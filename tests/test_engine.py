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


def test_two_connections_bridge_with_warning():
    con1, con2 = duckdb.connect(), duckdb.connect()
    con1.execute("CREATE TABLE a AS SELECT 1 AS k, 'one' AS va")
    con2.execute("CREATE TABLE b AS SELECT 1 AS k, 'two' AS vb")
    with pytest.warns(UserWarning, match="streaming"):
        out = d.from_duckdb(con1, "a").inner_join(
            d.from_duckdb(con2, "b"), on=col.k).collect()
    assert out.to_dicts() == [{"k": 1, "va": "one", "vb": "two"}]


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
    with pytest.raises(d.DpyrError, match="does not apply to parquet"):
        f.write(str(tmp_path / "x.parquet"), "t")
    with pytest.raises(d.DpyrError, match="can't infer a format"):
        f.write(str(tmp_path / "x.docx"))
    with pytest.raises(d.DpyrError, match="can't infer a format"):
        d.read(str(tmp_path / "x.docx"))
    with pytest.raises(d.DpyrError, match="does not apply to csv"):
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
    with pytest.raises(d.DpyrError, match="only applies to database"):
        d.read(data, "t")
    with pytest.raises(d.DpyrError, match="doesn't know what to do"):
        d.read(42)


# -- array / tensor / dataset interop --------------------------------------------

def test_read_numpy_arrays():
    import numpy as np
    one_d = d.read(np.array([1.5, 2.5, 3.5]))
    assert one_d.columns == ["value"] and one_d.collect().height == 3
    two_d = d.read(np.array([[1, 2], [3, 4], [5, 6]]))
    assert two_d.columns == ["column_0", "column_1"]
    assert two_d.collect()["column_1"].to_list() == [2, 4, 6]
    with pytest.raises(d.DpyrError, match="1-D or 2-D"):
        d.read(np.zeros((2, 2, 2)))
    # numpy arrays as dict values also work
    assert d.read({"x": np.array([1, 2])}).collect().height == 2


def test_read_huggingface_dataset():
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"text": ["hi", "yo"], "label": [0, 1]})
    f = d.read(ds)
    assert f.columns == ["text", "label"]
    out = f.filter(col.label == 1).collect()
    assert out["text"].to_list() == ["yo"]
    dd = datasets.DatasetDict({"train": ds, "test": ds})
    with pytest.raises(d.DpyrError, match="splits \\['train', 'test'\\]"):
        d.read(dd)
    assert d.read(dd, "train").collect().height == 2
    with pytest.raises(d.ColumnNotFoundError, match="Did you mean 'train'"):
        d.read(dd, "trian")


def test_to_numpy_roundtrip():
    f = d.read({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    arr = f.to_numpy()
    assert arr.shape == (2, 2) and arr[1, 1] == 4.0
    back = d.read(arr)
    assert back.collect().to_dicts()[0] == {"column_0": 1.0, "column_1": 3.0}


def test_read_torch_and_jax_if_available():
    torch = pytest.importorskip("torch")
    t = d.read(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert t.collect()["column_0"].to_list() == [1.0, 3.0]
    assert d.read({"x": [1.0]}).to_torch().shape == (1, 1)


def test_new_formats_roundtrip(tmp_path):
    f = d.from_dict({"g": ["a", "b"], "x": [1.5, 2.5]})
    for ext in ("json", "jsonl", "ndjson", "tsv", "xlsx"):
        p = str(tmp_path / f"t.{ext}")
        f.write(p)
        back = d.read(p)
        assert back.collect().to_dicts() == f.collect().to_dicts(), ext


def test_xlsx_named_sheet_and_bad_sheet_error(tmp_path):
    pytest.importorskip("fastexcel")
    f = d.from_dict({"g": ["a", "b"], "x": [1.5, 2.5]})
    p = str(tmp_path / "t.xlsx")
    f.write(p, "plots")
    assert d.read(p, "plots").collect().to_dicts() == f.collect().to_dicts()
    with pytest.raises(d.DpyrError, match=r"no sheet named 'oops'.*plots"):
        d.read(p, "oops")
    with pytest.raises(d.DpyrError, match="no such file"):
        d.read(str(tmp_path / "missing.xlsx"))


def test_xlsx_multi_sheet_workbook(tmp_path):
    pytest.importorskip("fastexcel")
    plots = d.from_dict({"plot": ["n", "s"], "acres": [3.2, 1.8]})
    notes = d.from_dict({"note": ["dry spring"]})
    p = str(tmp_path / "report.xlsx")
    plots.write(p, "plots")
    with pytest.warns(UserWarning, match="keeping the workbook's other"):
        notes.write(p, "notes")  # appends, preserving the plots sheet

    wb = d.read(p)
    assert isinstance(wb, d.Workbook)
    assert wb.sheets == ["plots", "notes"]  # original tab order kept
    assert wb.plots.collect().to_dicts() == plots.collect().to_dicts()
    assert wb["notes"].collect().to_dicts() == notes.collect().to_dicts()
    assert "plots" in repr(wb) and "notes" in dir(wb)
    with pytest.raises(d.DpyrError, match="Did you mean 'plots'"):
        wb.plotz
    # the two sheets must never share a cache entry
    assert d.read(p, "plots").collect().columns != d.read(p, "notes").collect().columns

    # rewriting an existing sheet replaces it in place
    plots2 = d.from_dict({"plot": ["e"], "acres": [9.9]})
    with pytest.warns(UserWarning):
        plots2.write(p, "plots")
    wb2 = d.read(p)
    assert wb2.sheets == ["plots", "notes"]
    assert wb2.plots.collect().to_dicts() == plots2.collect().to_dicts()
    assert wb2.notes.collect().to_dicts() == notes.collect().to_dicts()


def test_google_sheet_url_reads_as_workbook(tmp_path, monkeypatch):
    pytest.importorskip("fastexcel")
    import io
    import urllib.request

    plots = d.from_dict({"plot": ["n", "s"], "acres": [3.2, 1.8]})
    notes = d.from_dict({"note": ["dry spring"]})
    p = str(tmp_path / "wb.xlsx")
    plots.write(p, "plots")
    with pytest.warns(UserWarning):
        notes.write(p, "notes")
    payload = open(p, "rb").read()

    captured = {}

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req):
        captured["url"] = req.full_url
        return FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    url = "https://docs.google.com/spreadsheets/d/abc-123_XY/edit?gid=0"
    wb = d.read(url)
    assert isinstance(wb, d.Workbook)
    assert captured["url"] == (
        "https://docs.google.com/spreadsheets/d/abc-123_XY/export?format=xlsx")
    assert wb.sheets == ["plots", "notes"]
    assert url in repr(wb)
    assert d.read(url, "plots").collect().to_dicts() == plots.collect().to_dicts()
    with pytest.raises(d.DpyrError, match="Did you mean 'plots'"):
        d.read(url, "plotz")

    # a private sheet answers with the login page, not an xlsx zip
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req: FakeResponse(b"<html>sign in</html>"))
    with pytest.raises(d.DpyrError, match="not link-readable"):
        d.read(url)


def test_csv_gz_reads(tmp_path):
    import gzip
    p = str(tmp_path / "t.csv.gz")
    with gzip.open(p, "wt") as fh:
        fh.write("x,y\n1,a\n2,b\n")
    out = d.read(p).collect()
    assert out["x"].to_list() == [1, 2] and out["y"].to_list() == ["a", "b"]


def test_in_engine_json_copy(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM range(3) r(x)")
    p = str(tmp_path / "out.jsonl")
    d.from_duckdb(con, "t").mutate(y=col.x * 2).write(p)
    assert d.read(p).collect()["y"].to_list() == [0, 2, 4]


def test_sqlite_via_duckdb_scanner(tmp_path):
    import sqlite3
    p = str(tmp_path / "legacy.sqlite")
    sq = sqlite3.connect(p)
    sq.execute("CREATE TABLE users (name TEXT, age INTEGER)")
    sq.executemany("INSERT INTO users VALUES (?, ?)",
                   [("ana", 34), ("bo", 51)])
    sq.commit()
    sq.close()
    try:
        db = d.read(p)
    except d.DpyrError as e:
        pytest.skip(f"sqlite extension unavailable: {e}")
    assert "users" in db.tables
    out = d.read(p, "users").filter(col.age > 40).collect()
    assert out["name"].to_list() == ["bo"]


def test_read_write_accept_pathlib(tmp_path):
    f = d.from_dict({"x": [1, 2]})
    f.write(tmp_path / "t.parquet")          # a pathlib.Path, not a str
    assert d.read(tmp_path / "t.parquet").collect().height == 2


def test_join_anything_to_anything(tmp_path):
    """The full matrix: any pair of sources joins."""
    import sqlite3
    base = {"k": [1, 2], "v": ["a", "b"]}
    other = {"k": [1, 2], "w": [10, 20]}

    d.read(base).write(tmp_path / "a.csv")
    d.read(other).write(tmp_path / "b.parquet")
    d.read(other).write(tmp_path / "c.db", "t")
    d.read(other).write(tmp_path / "d.duckdb", "t")
    sq = sqlite3.connect(tmp_path / "e.sqlite")
    sq.execute("CREATE TABLE t (k INTEGER, w INTEGER)")
    sq.executemany("INSERT INTO t VALUES (?, ?)", [(1, 10), (2, 20)])
    sq.commit()
    sq.close()

    left_sources = [
        d.read(base),                          # in-memory
        d.read(tmp_path / "a.csv"),            # csv scan
        d.read(tmp_path / "c.db", "t"),        # duckdb file (note: has w not v)
    ]
    right_sources = [
        d.read(tmp_path / "b.parquet"),        # parquet scan
        d.read(tmp_path / "d.duckdb", "t"),    # a second duckdb file
        d.read(tmp_path / "e.sqlite", "t"),    # sqlite via scanner
        d.read(other),                         # in-memory
    ]
    import warnings
    for left in left_sources:
        for right in right_sources:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # cross-connection streams
                out = (left.inner_join(right, on=col.k, suffix=("_l", "_r"))
                       .arrange(col.k).collect())
            assert out.height == 2 and out["k"].to_list() == [1, 2]
