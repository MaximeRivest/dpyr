"""read(list of records) and nested columns carried through verbs (S35)."""

from __future__ import annotations

import dataclasses
import datetime as dtm
from typing import NamedTuple

import duckdb
import polars as pl
import pytest

import dpyr as d
from dpyr import col, lag, read

# -- read(records) -------------------------------------------------------------


def test_records_are_rows_with_bind_rows_key_union():
    f = read([{"a": 1, "b": "x"}, {"b": "y", "c": True}])
    assert f.columns == ["a", "b", "c"]
    assert f.collect().to_dicts() == [
        {"a": 1, "b": "x", "c": None}, {"a": None, "b": "y", "c": True}]


def test_records_scan_every_row_not_a_sample():
    # polars' own dict ingest samples the first rows; a key or a float that
    # first appears late must not be dropped or truncated
    rows = [{"a": 1, "v": [1]} for _ in range(300)] + [{"a": 2.5, "v": [1.5], "late": "x"}]
    out = read(rows).collect()
    assert out.schema["a"] == pl.Float64
    assert out["v"][-1].to_list() == [1.5]
    assert out["late"][-1] == "x"


def test_records_accept_dataclasses_namedtuples_and_tuples_of_rows():
    @dataclasses.dataclass
    class R:
        a: int
        b: str

    class N(NamedTuple):
        a: int
        b: str

    assert read([R(1, "x"), N(2, "y")]).collect().to_dicts() == [
        {"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert read(({"a": 1},)).shape == (1, 1)


def test_records_accept_pydantic_models():
    pydantic = pytest.importorskip("pydantic")

    class M(pydantic.BaseModel):
        a: int
        tags: list[str]

    f = read([M(a=1, tags=["x"])])
    assert f.schema == {"a": d.INT64, "tags": d.dtypes.nested("List(Str)")}


def test_records_combine_like_vctrs():
    assert read([{"a": True}, {"a": 2}]).collect()["a"].to_list() == [1, 2]
    assert read([{"a": 1}, {"a": 2.5}]).schema == {"a": d.FLOAT64}
    got = read([{"t": dtm.date(2020, 1, 1)}, {"t": dtm.datetime(2020, 1, 2, 3)}])
    assert got.schema == {"t": d.DATETIME}
    assert got.collect()["t"].to_list() == [
        dtm.datetime(2020, 1, 1), dtm.datetime(2020, 1, 2, 3)]


@pytest.mark.parametrize("bad, message", [
    ([], "no rows"),
    ([1, 2], "item 0 is of type int"),
    ([{"a": 1}, {"a": "x"}], "column 'a' mixes int .row 0. and str .row 1."),
    ([{"a": [1]}, {"a": ["x"]}], "column 'a' can't be stored"),
    ([{"a": object()}], "column 'a' holds a object in row 0"),
    ([{1: "x"}], "must be strings"),
])
def test_records_errors_name_the_problem(bad, message):
    with pytest.raises(d.DpyrError, match=message):
        read(bad)


def test_records_reject_table_argument():
    with pytest.raises(d.DpyrError, match="table="):
        read([{"a": 1}], "x")


# -- nested columns --------------------------------------------------------------

NESTED = pl.DataFrame({
    "k": [1, 2, 3],
    "tags": [["a", "b"], [], None],
    "usage": [{"inp": 3, "out": 4}, {"inp": 1, "out": None}, None],
})


def _both() -> list[d.DFrame]:
    from conftest import make_duckdb, make_polars
    return [make_polars(NESTED), make_duckdb(NESTED)]


def test_nested_dtypes_are_named_the_same_on_both_engines():
    schemas = [f.schema for f in _both()]
    assert schemas[0] == schemas[1] == {
        "k": d.INT64,
        "tags": d.dtypes.nested("List(Str)"),
        "usage": d.dtypes.nested("Struct(inp: Int64, out: Int64)"),
    }


def test_nested_inner_types_are_canonicalized():
    f = d.from_polars(pl.DataFrame({"v": [[1, 2]]}, schema={"v": pl.List(pl.Int32)}))
    assert f.schema["v"] == d.dtypes.nested("List(Int64)")
    assert f.collect().schema["v"] == pl.List(pl.Int64)


def test_duckdb_arrays_and_maps_are_nested():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT [1.5, 2.5]::FLOAT[2] AS emb, "
                "map(['a'], [1]) AS m, 1 AS k")
    f = read(con, "t")
    assert f.schema["emb"] == d.dtypes.nested("Array(Float64, 2)")
    assert f.schema["m"].nested
    assert f.collect()["emb"].to_list() == [[1.5, 2.5]]


@pytest.mark.parametrize("i", [0, 1])
def test_nested_columns_ride_along_through_verbs(i):
    f = _both()[i]
    other = read({"k": [1, 2, 3], "w": ["x", "y", "z"]})
    out = (f.filter(col.k >= 2)
           .mutate(t2=col.tags, prev=lag(col.tags))
           .left_join(other, on=col.k)
           .arrange(d.desc(col.k))
           .select(col.k, col.tags, col.usage, col.t2, col.prev, col.w)
           .collect())
    assert out["k"].to_list() == [3, 2]
    assert out["tags"].to_list() == [None, []]
    assert out["t2"].to_list() == [None, []]
    assert out["prev"].to_list() == [[], None]
    assert out["usage"].to_list() == [None, {"inp": 1, "out": None}]


@pytest.mark.parametrize("i", [0, 1])
def test_nested_first_last_and_is_na(i):
    f = _both()[i]
    out = f.arrange(col.k).summarize(first=col.tags.first(), n_na=col.tags.is_na().sum())
    assert out.collect().to_dicts() == [{"first": ["a", "b"], "n_na": 1}]


@pytest.mark.parametrize("build", [
    lambda f: f.arrange(col.tags),
    lambda f: f.group_by(col.tags),
    lambda f: f.distinct(),
    lambda f: f.filter(col.tags == "a"),
    lambda f: f.summarize(m=col.usage.max()),
    lambda f: f.mutate(r=d.min_rank(col.tags)),
    lambda f: f.left_join(f.select(col.k, col.tags), on=col.tags),
    lambda f: f.pivot_longer([col.tags]),
])
def test_nested_columns_are_not_compared(build):
    with pytest.raises(d.ExprTypeError, match="S35"):
        build(_both()[0])


def test_nested_typed_proxy_blocks_scalar_methods():
    f = _both()[0]
    assert isinstance(f.c.tags, d.NestedExpr)
    with pytest.raises(d.ExprTypeError, match="not available"):
        f.c.tags.str_len()


def test_where_is_nested_selects_them():
    assert _both()[0].select(-d.where(d.is_nested)).columns == ["k"]


def test_nested_round_trips_parquet_and_jsonl_but_not_csv(tmp_path):
    for f in _both():
        f.write(str(tmp_path / "x.parquet"))
        assert read(str(tmp_path / "x.parquet")).schema == f.schema
        f.write(str(tmp_path / "x.jsonl"))
        assert read(str(tmp_path / "x.jsonl")).schema["tags"] == f.schema["tags"]
        with pytest.raises(d.DpyrError, match="CSV can't hold nested"):
            f.write(str(tmp_path / "x.csv"))


# -- in-engine writes don't leak helper columns ----------------------------------------

def test_duckdb_writes_and_tables_carry_only_the_plan_columns(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (2, 'b'), (1, 'a')) v(k, v)")
    f = read(con, "t").arrange(col.k).mutate(t2=col.v)
    f.write(str(tmp_path / "x.parquet"))
    assert pl.read_parquet(tmp_path / "x.parquet").columns == ["k", "v", "t2"]
    f.to_table("out", con)
    assert [r[0] for r in con.execute("DESCRIBE out").fetchall()] == ["k", "v", "t2"]
    assert [r[0] for r in con.execute("SELECT k FROM out").fetchall()] == [1, 2]
