"""unnest_longer / unnest_wider / unnest (tidyr), on both engines.

Expected values are tidyr 1.3.2's output on the same data (checked by hand,
2026-10-03): list-columns of records are R's list of named lists.
"""

from __future__ import annotations

import dataclasses

import polars as pl
import pytest

import dpyr as d
from dpyr import col, desc, read
from conftest import make_duckdb, make_polars

ANALYSES = pl.DataFrame({
    "id": [1, 2, 3, 4],
    "a": [[{"q": "x", "n": 1}, {"q": "y", "n": 2}], [], None, [{"q": "z", "n": None}]],
    "z": ["p", "q", "r", "s"],
})


@dataclasses.dataclass
class Item:
    name: str
    size: int


@pytest.fixture(params=["polars", "duckdb"])
def frame(request):
    return (make_polars if request.param == "polars" else make_duckdb)(ANALYSES)


def test_unnest_longer_drops_empty_and_missing_lists(frame):
    out = frame.unnest_longer(col.a)
    assert out.schema == {"id": d.INT64, "a": d.dtypes.nested("Struct(q: Str, n: Int64)"), "z": d.STR}
    assert out.to_dicts() == [
        {"id": 1, "a": {"q": "x", "n": 1}, "z": "p"},
        {"id": 1, "a": {"q": "y", "n": 2}, "z": "p"},
        {"id": 4, "a": {"q": "z", "n": None}, "z": "s"},
    ]


def test_unnest_longer_keep_empty_keeps_one_null_row(frame):
    out = frame.unnest_longer("a", keep_empty=True)
    assert [r["id"] for r in out.to_dicts()] == [1, 1, 2, 3, 4]
    assert [r["a"] for r in out.to_dicts()][2:4] == [None, None]


def test_unnest_wider_puts_fields_in_place(frame):
    out = frame.unnest_longer(col.a).unnest_wider(col.a)
    assert out.columns == ["id", "q", "n", "z"]
    assert out.schema == {"id": d.INT64, "q": d.STR, "n": d.INT64, "z": d.STR}
    assert out.to_dicts() == [
        {"id": 1, "q": "x", "n": 1, "z": "p"},
        {"id": 1, "q": "y", "n": 2, "z": "p"},
        {"id": 4, "q": "z", "n": None, "z": "s"},
    ]


def test_unnest_is_longer_then_wider(frame):
    assert frame.unnest(col.a).to_dicts() == frame.unnest_longer(col.a).unnest_wider(col.a).to_dicts()


def test_unnest_wider_missing_struct_gives_nulls(make):
    f = make(pl.DataFrame({"id": [1, 2], "s": [{"q": "x", "n": 1}, None]}))
    assert f.unnest_wider(col.s).to_dicts() == [{"id": 1, "q": "x", "n": 1}, {"id": 2, "q": None, "n": None}]


def test_unnest_wider_name_clash_is_an_error_names_sep_fixes_it(frame):
    clash = frame.unnest_longer(col.a).mutate(q=d.lit(1))
    with pytest.raises(d.errors.DuplicateColumnError, match="names_sep"):
        clash.unnest_wider(col.a)
    assert clash.unnest_wider(col.a, names_sep="_").columns == ["id", "a_q", "a_n", "z", "q"]


def test_unnest_keeps_the_row_order_after_arrange(frame):
    out = frame.arrange(desc(col.id)).unnest(col.a)
    assert [(r["id"], r["q"]) for r in out.to_dicts()] == [(4, "z"), (1, "x"), (1, "y")]


def test_unnest_then_verbs_and_groups(frame):
    out = (frame.group_by(col.z).unnest(col.a)
           .summarize(analyses=d.n(), total=col.n.sum()))
    assert out.to_dicts() == [{"z": "p", "analyses": 2, "total": 3}, {"z": "s", "analyses": 1, "total": 0}]


def test_unnest_of_plain_values_only_gets_longer(make):
    f = make(pl.DataFrame({"k": [1, 2], "tags": [["a", "b"], ["c"]]}))
    out = f.unnest(col.tags)
    assert out.schema == {"k": d.INT64, "tags": d.STR}
    assert out.to_dicts() == [{"k": 1, "tags": "a"}, {"k": 1, "tags": "b"}, {"k": 2, "tags": "c"}]


def test_unnest_longer_of_an_array():
    import duckdb
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS k, [1.5, 2.5]::DOUBLE[2] AS emb")
    for f in (read(con, "t"), read(read(con, "t").to_polars())):
        assert f.unnest_longer(col.emb).to_dicts() == [{"k": 1, "emb": 1.5}, {"k": 1, "emb": 2.5}]


def test_nested_lists_unnest_one_level_at_a_time(make):
    f = make(pl.DataFrame({"k": [1], "m": [[[1, 2], [3]]]}))
    once = f.unnest_longer(col.m)
    assert once.schema["m"] == d.dtypes.nested("List(Int64)")
    assert once.unnest_longer(col.m).to_dicts() == [{"k": 1, "m": 1}, {"k": 1, "m": 2}, {"k": 1, "m": 3}]


def test_wrong_column_types_are_refused_on_the_line(frame):
    with pytest.raises(d.errors.ExprTypeError, match="needs a list column"):
        frame.unnest_longer(col.z)
    with pytest.raises(d.errors.ExprTypeError, match="unnest_longer\\(\\) it first"):
        frame.unnest_wider(col.a)
    with pytest.raises(d.errors.ColumnNotFoundError, match="Did you mean 'a'"):
        frame.unnest_longer("aa")


def test_a_row_function_returning_records_unnests():
    # the AI-function case: one call per row returns a list of records
    @d.vectorize
    def items(text: str) -> list[Item]:
        return [Item(w, len(w)) for w in text.split()]

    f = read({"doc": [1, 2], "text": ["a bb", ""]})
    out = f.mutate(items=items(col.text)).unnest(col.items)
    assert out.to_dicts() == [{"doc": 1, "text": "a bb", "name": "a", "size": 1},
                              {"doc": 1, "text": "a bb", "name": "bb", "size": 2}]
    assert out.slice_head(1).to_dicts() == out.to_dicts()[:1]


def test_struct_field_names_read_back_from_the_dtype():
    t = d.dtypes.nested("Struct(a: List(Struct(b: Str, c: Int64)), d: Float64)")
    assert d.dtypes.struct_fields(t) == [("a", d.dtypes.nested("List(Struct(b: Str, c: Int64))")),
                                         ("d", d.FLOAT64)]
    assert d.dtypes.list_inner(d.dtypes.nested("Array(Float64, 3)")) == d.FLOAT64
    with pytest.raises(ValueError):
        d.dtypes.struct_fields(d.dtypes.nested("Struct(odd: name: Str)"))
