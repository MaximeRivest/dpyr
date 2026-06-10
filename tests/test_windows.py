"""1.1.0 feature tests: window functions, slice_min/max, separate/unite,
relocate, coalesce/replace_na — on both backends."""

from __future__ import annotations

import pytest

import dpyr as d
from dpyr import (
    coalesce,
    col,
    cum_max,
    cum_min,
    cum_sum,
    dense_rank,
    desc,
    lag,
    lead,
    min_rank,
    percent_rank,
    replace_na,
    row_number,
)

DATA = {"g": ["a", "a", "a", "b", "b"], "x": [3, 1, 2, 9, None]}


def rows(f):
    return f.collect().to_dicts()


# -- lag / lead -----------------------------------------------------------

def test_lag_lead_ungrouped(make):
    out = make({"x": [1, 2, 3]}).mutate(
        p=lag(col.x), nx=lead(col.x), p2=lag(col.x, 2, default=0)).collect()
    assert out["p"].to_list() == [None, 1, 2]
    assert out["nx"].to_list() == [2, 3, None]
    assert out["p2"].to_list() == [0, 0, 1]


def test_lag_grouped_respects_groups(make):
    f = make(DATA).group_by(col.g).mutate(p=lag(col.x)).ungroup()
    by = {(r["g"], r["x"]): r["p"] for r in rows(f)}
    assert by[("a", 3)] is None  # first in its group
    assert by[("b", 9)] is None  # lag never crosses groups
    assert by[("a", 1)] == 3


def test_lag_respects_pending_arrange(make):
    f = make({"x": [3, 1, 2]}).arrange(col.x).mutate(p=lag(col.x))
    assert [(r["x"], r["p"]) for r in rows(f)] == [(1, None), (2, 1), (3, 2)]


# -- ranks ---------------------------------------------------------------

def test_row_number_and_ranks(make):
    out = make({"x": [30, 10, 10, 20]}).mutate(
        rn=row_number(), r=min_rank(col.x), dr=dense_rank(col.x),
        rd=min_rank(desc(col.x))).collect()
    assert out["rn"].to_list() == [1, 2, 3, 4]
    assert out["r"].to_list() == [4, 1, 1, 3]   # min_rank: ties share min
    assert out["dr"].to_list() == [3, 1, 1, 2]
    assert out["rd"].to_list() == [1, 3, 3, 2]


def test_rank_null_is_null(make):
    out = make({"x": [2, None, 1]}).mutate(r=min_rank(col.x)).collect()
    assert out["r"].to_list() == [2, None, 1]


def test_percent_rank_matches_dplyr_formula(make):
    out = make({"x": [1, 2, 3, None]}).mutate(pr=percent_rank(col.x)).collect()
    assert out["pr"].to_list()[:3] == [0.0, 0.5, 1.0]
    assert out["pr"].to_list()[3] is None


# -- cumulative -------------------------------------------------------------

def test_cumulatives(make):
    out = make({"x": [1, 3, None, 2]}).mutate(
        cs=cum_sum(col.x), cm=cum_min(col.x), cx=cum_max(col.x)).collect()
    assert out["cs"].to_list() == [1, 4, None, 6]  # S29
    assert out["cm"].to_list() == [1, 1, None, 1]
    assert out["cx"].to_list() == [1, 3, None, 3]


def test_cum_sum_grouped(make):
    f = make(DATA).group_by(col.g).mutate(cs=cum_sum(col.x)).ungroup()
    by = {(r["g"], r["x"]): r["cs"] for r in rows(f)}
    assert by[("a", 3)] == 3 and by[("a", 1)] == 4 and by[("a", 2)] == 6
    assert by[("b", 9)] == 9 and by[("b", None)] is None


def test_window_in_filter(make):
    f = make({"x": [5, 1, 3]}).filter(min_rank(col.x) <= 2)
    assert sorted(r["x"] for r in rows(f)) == [1, 3]


def test_window_in_summarize_rejected(make):
    with pytest.raises(d.ExprTypeError, match="mutate"):
        make(DATA).summarize(bad=lag(col.x))


# -- slice_min / slice_max ----------------------------------------------------

def test_slice_min_max_with_ties(make):
    f = make({"x": [10, 10, 20, 30]})
    assert [r["x"] for r in rows(f.slice_min(col.x))] == [10, 10]  # ties kept
    assert [r["x"] for r in rows(f.slice_min(col.x, with_ties=False))] == [10]
    assert [r["x"] for r in rows(f.slice_max(col.x, 2))] == [30, 20]


def test_slice_min_grouped(make):
    f = make(DATA).group_by(col.g).slice_min(col.x)
    out = sorted((r["g"], r["x"]) for r in rows(f.ungroup()))
    assert out == [("a", 1), ("b", 9)]


# -- separate / unite / relocate -------------------------------------------------

def test_separate(make):
    f = make({"id": [1, 2, 3], "code": ["a_x", "b_y_z", "c"]})
    out = rows(f.separate(col.code, into=["k", "v"]))
    assert out == [
        {"id": 1, "k": "a", "v": "x"},
        {"id": 2, "k": "b", "v": "y"},   # extra piece dropped, like tidyr
        {"id": 3, "k": "c", "v": None},  # missing piece -> null
    ]


def test_separate_keep_original(make):
    f = make({"code": ["a_x"]}).separate(col.code, into=["k", "v"], remove=False)
    assert f.columns == ["k", "v", "code"]


def test_unite(make):
    f = make({"a": ["x", None], "b": ["1", "2"], "z": [9, 8]})
    out = rows(f.unite("ab", [col.a, col.b]))
    assert out == [{"ab": "x_1", "z": 9}, {"ab": "NA_2", "z": 8}]  # S32
    out2 = rows(f.unite("ab", [col.a, col.b], na_rm=True))
    assert out2[1]["ab"] == "2"


def test_unite_position_and_keep(make):
    f = make({"z": [1], "a": ["x"], "b": ["y"]})
    united = f.unite("ab", [col.a, col.b])
    assert united.columns == ["z", "ab"]
    kept = f.unite("ab", [col.a, col.b], remove=False)
    assert kept.columns == ["z", "ab", "a", "b"]


def test_relocate(make):
    f = make({"a": [1], "b": [2], "c": [3]})
    assert f.relocate(col.c).columns == ["c", "a", "b"]
    assert f.relocate(col.a, after=col.c).columns == ["b", "c", "a"]
    assert f.relocate(col.c, before=col.b).columns == ["a", "c", "b"]


# -- coalesce / replace_na -------------------------------------------------------

def test_coalesce_and_replace_na(make):
    f = make({"x": [None, 2], "y": [10, None]})
    out = rows(f.mutate(z=coalesce(col.x, col.y, 0), w=replace_na(col.x, -1)))
    assert [r["z"] for r in out] == [10, 2]
    assert [r["w"] for r in out] == [-1, 2]
