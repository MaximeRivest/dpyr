"""tidyselect + across tests (Epic 6), executed on both backends."""

from __future__ import annotations

import pytest

import dpyr as d
from dpyr import (
    across,
    col,
    contains,
    ends_with,
    everything,
    is_numeric,
    is_string,
    matches,
    starts_with,
    where,
)

from conftest import STARWARS


def test_starts_ends_contains_matches(make):
    f = make(STARWARS)
    assert f.select(starts_with("ma")).columns == ["mass"]
    assert f.select(ends_with("s")).columns == ["mass", "species"]
    assert f.select(contains("eig")).columns == ["height"]
    assert f.select(matches("^[hm]")).columns == ["height", "mass"]


def test_where_and_everything(make):
    f = make(STARWARS)
    assert f.select(where(is_numeric)).columns == ["height", "mass"]
    assert f.select(where(is_string)).columns == ["name", "species"]
    assert f.select(everything()).columns == f.columns


def test_negation(make):
    f = make(STARWARS)
    assert f.select(-col.name).columns == ["height", "mass", "species"]
    assert f.select(-starts_with("ma")).columns == ["name", "height", "species"]
    assert f.select(col.name, -col.name) .columns == []


def test_mixed_selection_order(make):
    f = make(STARWARS)
    assert f.select(col.species, starts_with("h")).columns == ["species", "height"]


def test_unknown_column_in_selection(make):
    with pytest.raises(d.ColumnNotFoundError):
        make(STARWARS).select(col.heigth)


def test_across_summarize(make):
    out = make(STARWARS).summarize(
        across(where(is_numeric), "mean", names="{col}_mean")).collect()
    assert out.columns == ["height_mean", "mass_mean"]
    assert out["mass_mean"][0] == pytest.approx(67.0)


def test_across_multiple_fns(make):
    out = make(STARWARS).summarize(
        across(col.height, ["min", "max"])).collect()
    assert out.columns == ["height_min", "height_max"]
    assert out.to_dicts() == [{"height_min": 66, "height_max": 228}]


def test_across_with_lambda_and_dict(make):
    out = make(STARWARS).summarize(
        across(where(is_numeric), {"rng": lambda c: c.max() - c.min()},
               names="{col}_{fn}")).collect()
    assert out.to_dicts() == [{"height_rng": 162, "mass_rng": 95.0}]


def test_across_in_mutate(make):
    out = (make(STARWARS)
           .mutate(across(where(is_numeric), lambda c: c * 2))
           .collect())
    assert out["height"].to_list()[0] == 344  # overwritten in place


def test_across_mixed_with_kwargs(make):
    from dpyr import n
    out = (make(STARWARS).group_by(col.species)
           .summarize(across(col.height, "mean", names="mh"), n=n())
           .collect())
    assert out.columns == ["species", "mh", "n"]


def test_across_unknown_shortcut():
    with pytest.raises(d.ExprTypeError, match="unknown function shortcut"):
        across(everything(), "meen")


def test_pivot_longer_with_selector(make):
    f = make({"id": ["a"], "x_1": [1.0], "x_2": [2.0]})
    out = f.pivot_longer([starts_with("x_")], names_to="k").collect()
    assert out.columns == ["id", "k", "value"]
    assert out.height == 2
