"""1.2.0 tests: the arrow bridge, engine selection, in-engine outputs, and
the readr-style IO surface."""

from __future__ import annotations

import duckdb
import pytest

import dpyr as d
from dpyr import col, n

from conftest import STARWARS


def fresh_db(tmp_path, name="shop.db"):
    return str(tmp_path / name)


# -- the bridge ---------------------------------------------------------------

def test_mixed_join_runs_in_duckdb():
    con = duckdb.connect()
    con.execute("CREATE TABLE w AS SELECT * FROM (VALUES (1,'big'),(2,'small')) v(k,size)")
    db = d.from_duckdb(con, "w")
    mem = d.from_dict({"k": [1, 2, 3], "qty": [10, 20, 30]})
    out = mem.inner_join(db, on=col.k).arrange(col.k).collect()
    assert out.to_dicts() == [
        {"k": 1, "qty": 10, "size": "big"},
        {"k": 2, "qty": 20, "size": "small"}]
    # and the other direction
    out2 = db.left_join(mem, on=col.k).arrange(col.k).collect()
    assert out2["qty"].to_list() == [10, 20]


def test_bridge_unregisters_after_collect():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS k")
    mem = d.from_dict({"k": [1], "v": [9]})
    d.from_duckdb(con, "t").inner_join(mem, on=col.k).collect()
    leftover = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()
        if r[0].startswith("__dpyr_")]
    assert leftover == []


def test_two_connections_still_error():
    con1, con2 = duckdb.connect(), duckdb.connect()
    con1.execute("CREATE TABLE a AS SELECT 1 AS k")
    con2.execute("CREATE TABLE b AS SELECT 1 AS k")
    with pytest.raises(d.DpyrError, match="different duckdb connections"):
        d.from_duckdb(con1, "a").inner_join(
            d.from_duckdb(con2, "b"), on=col.k).collect()


# -- engine selection -----------------------------------------------------------

def test_engine_override_duckdb_on_memory_frame():
    f = d.from_dict({"x": [3, 1, 2]})
    a = f.arrange(col.x).collect()
    b = f.arrange(col.x).collect(engine="duckdb")
    assert a.equals(b)


def test_engine_polars_cannot_read_duckdb():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS x")
    with pytest.raises(d.DpyrError, match="cannot read duckdb"):
        d.from_duckdb(con, "t").collect(engine="polars")


def test_unified_sampling_across_engines():
    f = d.from_dict({"x": list(range(50))})
    a = f.slice_sample(7, seed=11).collect()["x"].to_list()
    b = f.slice_sample(7, seed=11).collect(engine="duckdb")["x"].to_list()
    assert a == b  # S33: same seed, same rows, both engines
    assert f.slice_sample(7, seed=12).collect()["x"].to_list() != a


# -- in-engine outputs ------------------------------------------------------------

def test_persist_duckdb_is_in_engine():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM range(5) r(x)")
    p = d.from_duckdb(con, "t").mutate(y=col.x * 2).persist()
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name LIKE 'dpyr_persist%'").fetchall()]
    assert tables, "persist should create a temp table"
    assert p.collect()["y"].to_list() == [0, 2, 4, 6, 8]


def test_to_table_and_to_view():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM range(4) r(x)")
    f = d.from_duckdb(con, "t").filter(col.x > 1)
    t = f.to_table("kept")
    assert t.collect()["x"].to_list() == [2, 3]
    assert con.execute("SELECT count(*) FROM kept").fetchone()[0] == 2
    v = f.mutate(y=col.x * 10).to_view("kept_v")
    assert v.collect()["y"].to_list() == [20, 30]
    # a view tracks its source table
    con.execute("INSERT INTO t VALUES (9)")
    d.cache_clear()
    assert 90 in v.collect()["y"].to_list()


def test_to_table_from_memory_needs_destination():
    f = d.from_dict({"x": [1]})
    with pytest.raises(d.DpyrError, match="duckdb destination"):
        f.to_table("nope")
    con = duckdb.connect()
    out = f.to_table("yep", con=con)
    assert out.collect()["x"].to_list() == [1]


def test_show_query_both_engines():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS x")
    sql = d.from_duckdb(con, "t").filter(col.x > 0).show_query()
    assert sql.startswith("SELECT") and "WHERE" in sql
    plan = d.from_dict({"x": [1]}).filter(col.x > 0).show_query()
    assert isinstance(plan, str) and plan  # polars explain output


def test_write_parquet_in_engine_and_polars(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM range(3) r(x)")
    p1 = str(tmp_path / "duck.parquet")
    d.from_duckdb(con, "t").mutate(y=col.x + 1).write_parquet(p1)
    assert d.read_parquet(p1).collect()["y"].to_list() == [1, 2, 3]
    p2 = str(tmp_path / "pol.parquet")
    d.from_dict({"x": [5, 6]}).write_parquet(p2)
    assert d.read_parquet(p2).collect()["x"].to_list() == [5, 6]


# -- readr surface -------------------------------------------------------------------

def test_write_read_duckdb_roundtrip(tmp_path):
    path = fresh_db(tmp_path)
    gold = (d.from_dict(STARWARS).group_by(col.species)
            .summarize(n=n(), mh=col.height.mean()))
    bound = gold.write_duckdb(path, "gold")
    assert bound.collect().height == 4
    db = d.read_duckdb(path)
    assert db.tables == ["gold"]
    assert db.gold.collect().height == 4
    assert d.read_duckdb(path, "gold").collect().height == 4
    assert "gold" in repr(db)
    assert "gold" in dir(db)


def test_read_duckdb_missing_table_suggests(tmp_path):
    path = fresh_db(tmp_path)
    d.from_dict({"x": [1]}).write_duckdb(path, "orders")
    db = d.read_duckdb(path)
    with pytest.raises(d.ColumnNotFoundError, match="Did you mean 'orders'"):
        db.orderz
    with pytest.raises(d.DpyrError, match="no such file"):
        d.read_duckdb(str(tmp_path / "ghost.db"))


def test_ipc_roundtrip_and_glimpse(tmp_path, capsys):
    path = str(tmp_path / "t.arrow")
    f = d.from_dict({"name": ["ana", None], "amount": [1.5, 2.5]})
    f.write_ipc(path)
    back = d.read_ipc(path)
    assert back.collect().to_dicts() == f.collect().to_dicts()
    result = back.glimpse()
    out = capsys.readouterr().out
    assert "Rows: 2" in out and "Columns: 2" in out
    assert "$ name" in out and "NA" in out and "<Float64>" in out
    assert result.columns == back.columns  # chains


def test_database_sql_escape_hatch(tmp_path):
    path = fresh_db(tmp_path)
    d.from_dict({"x": [1, 2, 3]}).write_duckdb(path, "t")
    db = d.read_duckdb(path)
    assert db.sql("SELECT sum(x) AS s FROM t").collect()["s"].to_list() == [6]


def test_write_duckdb_from_another_connection(tmp_path):
    """A frame bound to one duckdb connection can land in a database file
    (a different connection): dpyr copies via arrow instead of emitting SQL
    that references tables the destination doesn't have."""
    con = duckdb.connect()
    con.execute("CREATE TABLE src AS SELECT * FROM range(3) r(x)")
    f = d.from_duckdb(con, "src").mutate(y=col.x * 2)
    out = f.write_duckdb(str(tmp_path / "land.db"), "copied")
    assert out.collect()["y"].to_list() == [0, 2, 4]
    with pytest.raises(d.DpyrError, match="different duckdb connection"):
        from dpyr.io import _file_con
        f.to_view("nope", con=_file_con(str(tmp_path / "land.db")))


# -- the one reader / one writer ------------------------------------------------

def test_read_write_dispatch_by_extension(tmp_path):
    f = d.from_dict({"g": ["a", "b"], "x": [1.5, 2.5]})
    for ext in ("parquet", "csv", "arrow"):
        p = str(tmp_path / f"t.{ext}")
        f.write(p)
        back = d.read(p)
        assert back.collect().to_dicts() == f.collect().to_dicts(), ext
    bound = f.write(str(tmp_path / "shop.db"), "t")
    assert bound.collect().height == 2
    db = d.read(str(tmp_path / "shop.db"))
    assert isinstance(db, d.Database) and db.tables == ["t"]
    one = d.read(str(tmp_path / "shop.db"), "t")
    assert one.collect().height == 2


def test_read_write_errors_are_helpful(tmp_path):
    f = d.from_dict({"x": [1]})
    with pytest.raises(d.DpyrError, match="needs a table name"):
        f.write(str(tmp_path / "x.db"))
    with pytest.raises(d.DpyrError, match="only applies to duckdb"):
        f.write(str(tmp_path / "x.parquet"), "t")
    with pytest.raises(d.DpyrError, match="can't infer a format"):
        f.write(str(tmp_path / "x.xlsx"))
    with pytest.raises(d.DpyrError, match="can't infer a format"):
        d.read(str(tmp_path / "x.xlsx"))
    with pytest.raises(d.DpyrError, match="only applies to duckdb"):
        d.read(str(tmp_path / "x.csv"), "t")


def test_read_glob(tmp_path):
    import polars as pl
    pl.DataFrame({"x": [1]}).write_parquet(tmp_path / "a.parquet")
    pl.DataFrame({"x": [2]}).write_parquet(tmp_path / "b.parquet")
    out = d.read(str(tmp_path / "*.parquet"))
    assert sorted(out.collect()["x"].to_list()) == [1, 2]


def test_write_csv_in_engine(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM range(3) r(x)")
    p = str(tmp_path / "out.csv")
    d.from_duckdb(con, "t").mutate(y=col.x * 2).write_csv(p)
    assert d.read(p).collect()["y"].to_list() == [0, 2, 4]


def test_read_is_the_universal_ingest():
    import polars as pl
    import pyarrow as pa
    data = {"x": [1, 2], "g": ["a", "b"]}
    expected = d.read(data).collect().to_dicts()
    assert d.read(pl.DataFrame(data)).collect().to_dicts() == expected
    assert d.read(pl.LazyFrame(data)).collect().to_dicts() == expected
    assert d.read(pa.table(data)).collect().to_dicts() == expected
    import pandas as pd
    assert d.read(pd.DataFrame(data)).collect().to_dicts() == expected
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS x")
    db = d.read(con)
    assert isinstance(db, d.Database) and db.tables == ["t"]
    assert d.read(con, "t").collect()["x"].to_list() == [1]
    with pytest.raises(d.DpyrError, match="only applies to duckdb"):
        d.read(data, "t")
    with pytest.raises(d.DpyrError, match="doesn't know what to do"):
        d.read(42)
