"""Regression tests for the pre-1.0 adversarial review findings.

Each test names the finding it pins down; most were demonstrated live by
review agents before being fixed.
"""

from __future__ import annotations

import warnings

import duckdb
import pytest

import dpyr as d
from dpyr import across, col, desc, everything, is_numeric, n, plan_hash, where

from conftest import STARWARS


# -- cache-key completeness (blockers) ------------------------------------

def test_join_hash_distinguishes_na_matches_and_suffix(make):
    a = make({"k": [1, None], "v": [10, 20]})
    b = make({"k": [1, None], "w": [5, 6]})
    j1 = a.inner_join(b, on=col.k)
    j2 = a.inner_join(b, on=col.k, na_matches="never")
    j3 = a.left_join(b, on=col.k, suffix=("_l", "_r"))
    j4 = a.left_join(b, on=col.k)
    assert plan_hash(j1.plan) != plan_hash(j2.plan)
    assert plan_hash(j3.plan) != plan_hash(j4.plan)
    # warm-cache correctness: collect both, results must differ
    assert len(j1.collect()) == 2  # NA matches NA (S10)
    assert len(j2.collect()) == 1


def test_grouped_and_ungrouped_plans_hash_differently(make):
    f = make({"g": ["a", "a", "b"], "x": [1.0, 2.0, 30.0]})
    g = f.group_by(col.g)
    grouped = g.summarize(m=col.x.mean())
    ungrouped = g.ungroup().summarize(m=col.x.mean())
    assert plan_hash(grouped.plan) != plan_hash(ungrouped.plan)
    assert grouped.collect().height == 2
    out = ungrouped.collect()
    assert out.height == 1 and out["m"][0] == pytest.approx(11.0)


# -- duckdb order semantics (blockers) --------------------------------------

def test_arrange_respected_by_first_last_in_summarize(make):
    f = make({"g": ["a", "a", "a", "b", "b"], "x": [3, 1, 2, 9, 7]})
    out = (f.arrange(col.x).group_by(col.g)
           .summarize(top=col.x.first(), bottom=col.x.last()).collect())
    assert out["top"].to_list() == [1, 7]
    assert out["bottom"].to_list() == [3, 9]


def test_arrange_respected_by_first_in_grouped_mutate(make):
    f = make({"g": ["a", "a", "b"], "x": [3, 1, 9]})
    out = (f.arrange(col.x).group_by(col.g).mutate(top=col.x.first())
           .ungroup().arrange(col.g, col.x).collect())
    assert out["top"].to_list() == [1, 1, 9]


def test_arrange_respected_by_ungrouped_first(make):
    f = make({"x": [3, 1, 9]})
    out = f.arrange(desc(col.x)).summarize(top=col.x.first()).collect()
    assert out["top"][0] == 9


# -- ungrouped window aggregates on duckdb (major) ----------------------------

def test_ungrouped_mutate_with_aggregate_broadcasts(make):
    f = make({"x": [1.0, 2.0, 3.0]})
    out = f.mutate(m=col.x.mean(), pct=col.x / col.x.sum(), cnt=n()).collect()
    assert out["m"].to_list() == [2.0, 2.0, 2.0]
    assert out["pct"].to_list() == pytest.approx([1 / 6, 2 / 6, 3 / 6])
    assert out["cnt"].to_list() == [3, 3, 3]


def test_ungrouped_filter_with_aggregate(make):
    f = make({"x": [1, 5, 3]})
    out = f.filter(col.x == col.x.max()).collect()
    assert out["x"].to_list() == [5]


# -- grouped-frame verb coverage (majors) ------------------------------------

def test_grouped_count_counts_within_groups(make):
    f = make({"g": ["a", "a", "b", "b"], "x": [1, 1, 2, 3]})
    out = f.group_by(col.g).count(col.x)
    assert isinstance(out, d.GroupedDFrame)
    assert out.ungroup().collect().to_dicts() == [
        {"g": "a", "x": 1, "n": 2}, {"g": "b", "x": 2, "n": 1},
        {"g": "b", "x": 3, "n": 1}]


def test_grouped_join_and_pivot_longer_work(make):
    g = make(STARWARS).group_by(col.species)
    other = make({"name": ["Luke"], "film": ["ANH"]})
    joined = g.inner_join(other, on=col.name)
    assert isinstance(joined, d.GroupedDFrame)
    assert len(joined.ungroup()) == 1
    longer = make({"g": ["a"], "x": [1.0], "y": [2.0]}).group_by(col.g) \
        .pivot_longer([col.x, col.y])
    assert len(longer.ungroup()) == 2


def test_persist_preserves_grouping(make):
    g = make({"g": ["a", "a", "b"], "x": [1.0, 2.0, 30.0]}).group_by(col.g)
    direct = g.summarize(m=col.x.mean()).collect()
    persisted = g.persist().summarize(m=col.x.mean()).collect()
    assert direct.equals(persisted)


# -- across + groups (major) -------------------------------------------------

def test_across_excludes_grouping_columns(make):
    f = make({"g": [1, 1, 2], "x": [10.0, 20.0, 30.0]})
    out = f.group_by(col.g).summarize(across(where(is_numeric), "mean"))
    assert out.collect().to_dicts() == [{"g": 1, "x": 15.0}, {"g": 2, "x": 30.0}]
    m = f.group_by(col.g).mutate(across(everything(), lambda c: c.max(),
                                        names="{col}_max"))
    assert "g_max" not in m.columns and "x_max" in m.columns


# -- pivot_wider edge cases (major/minor) -------------------------------------

def test_pivot_wider_duplicates_warn(make):
    f = make({"id": [1, 1, 2], "k": ["a", "a", "a"], "v": [10, 99, 30]})
    with pytest.warns(UserWarning, match="not uniquely identified"):
        out = f.pivot_wider(names_from=col.k, values_from=col.v)
    assert out.collect()["a"].to_list() == [10, 30]


def test_pivot_wider_without_id_columns(make):
    f = make({"k": ["a", "b"], "v": [1.0, 2.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = f.pivot_wider(names_from=col.k, values_from=col.v)
    assert out.collect().to_dicts() == [{"a": 1.0, "b": 2.0}]


# -- backend safety (major) ----------------------------------------------------

def test_cross_connection_join_raises_clearly():
    con1, con2 = duckdb.connect(), duckdb.connect()
    con1.execute("CREATE TABLE t1 AS SELECT 1 AS k")
    con2.execute("CREATE TABLE t2 AS SELECT 1 AS k, 'x' AS v")
    a = d.from_duckdb(con1, "t1")
    b = d.from_duckdb(con2, "t2")
    with pytest.raises(d.DpyrError, match="different duckdb connections"):
        a.inner_join(b, on=col.k).collect()


# -- cache hygiene (minors) ------------------------------------------------------

def test_seedless_sample_not_frozen_by_cache(make):
    f = make({"x": list(range(1000))})
    runs = {tuple(f.slice_sample(5).collect()["x"].to_list()) for _ in range(5)}
    assert len(runs) > 1  # would be 1 if the cache froze the sample


def test_reread_file_hits_same_plan_hash(tmp_path):
    import polars as pl
    path = str(tmp_path / "t.parquet")
    pl.DataFrame({"x": [1, 2]}).write_parquet(path)
    h1 = plan_hash(d.read_parquet(path).plan)
    h2 = plan_hash(d.read_parquet(path).plan)
    assert h1 == h2  # content-addressed token: notebook re-runs share cache


def test_duckdb_source_token_stable_per_connection():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS x")
    assert plan_hash(d.from_duckdb(con, "t").plan) == \
        plan_hash(d.from_duckdb(con, "t").plan)
