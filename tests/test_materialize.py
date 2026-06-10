"""Materialization model tests (Epic 3): display-eager boundaries, the
plan-hash cache, persist checkpoints, and the lazy opt-out."""

from __future__ import annotations

from polars.testing import assert_frame_equal

import dpyr as d
from dpyr import col, n

from conftest import STARWARS


def test_len_shape_iter_getitem(make):
    f = make(STARWARS)
    assert len(f) == 6
    assert f.shape == (6, 4)
    assert [r["name"] for r in f][0] == "Luke"
    assert f["height"] == STARWARS["height"]


def test_repr_is_display_eager_with_provenance(make, backend):
    f = make(STARWARS).filter(col.height > 100)
    r = repr(f)
    assert f"source: {backend}" in r
    assert "Luke" in r  # actual rows shown
    assert "showing" in r


def test_lazy_opt_out_never_shows_rows(make):
    f = make(STARWARS).lazy()
    r = repr(f)
    assert "Luke" not in r and "(lazy)" in r
    # eager() flips back
    assert "Luke" in repr(f.eager())


def test_global_interactive_option(make):
    f = make(STARWARS)
    d.options.interactive = False
    try:
        assert "Luke" not in repr(f)
    finally:
        d.options.interactive = True


def test_collect_is_cached(make):
    d.cache_clear()
    f = make(STARWARS).filter(col.mass > 50)
    a = f.collect()
    assert d.cache_size() >= 1
    b = f.collect()
    assert a is b  # same object: served from cache


def test_persist_equals_direct_collect(make):
    """Metamorphic: collect-once == persist-stepwise (TESTING.md)."""
    f = make(STARWARS)
    chain = (f.filter(col.height > 80)
              .mutate(bmi=col.mass / (col.height / 100) ** 2)
              .group_by(col.species).summarize(n=n(), mb=col.bmi.mean()))
    direct = chain.collect()
    stepped = (f.filter(col.height > 80).persist()
                .mutate(bmi=col.mass / (col.height / 100) ** 2).persist()
                .group_by(col.species).summarize(n=n(), mb=col.bmi.mean())
                .collect())
    assert_frame_equal(direct, stepped)


def test_persist_stays_on_backend(make, backend):
    f = make(STARWARS).filter(col.height > 100).persist()
    assert f._provenance() == backend  # duckdb persists to a temp table
    assert len(f) == 4


def test_repr_then_collect_consistent(make):
    f = make(STARWARS).arrange(col.height)
    first_shown = repr(f)
    collected = f.collect()
    assert collected["name"][0] == "Yoda"
    assert "Yoda" in first_shown


def test_schema_only_frames_never_execute():
    f = d.from_schema({"x": d.INT64}).mutate(y=col.x + 1)
    assert "schema-only" in repr(f)
    try:
        f.collect()
        raise AssertionError("expected BackendError")
    except d.DpyrError as e:
        assert "no data attached" in str(e)


def test_pivot_wider_is_implicitly_persisted(make):
    data = {"id": ["a", "a", "b", "b"], "key": ["x", "y", "x", "y"],
            "val": [1.0, 2.0, 3.0, 4.0]}
    wide = make(data).pivot_wider(names_from=col.key, values_from=col.val)
    # schema known immediately because the input was materialized
    assert wide.columns == ["id", "x", "y"]
    assert wide.schema["x"] == d.FLOAT64
