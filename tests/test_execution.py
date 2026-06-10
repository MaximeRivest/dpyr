"""Per-verb execution tests, parameterized over BOTH backends.

Every test runs identically on polars and duckdb — the `make` fixture
builds the same data on whichever backend is active.
"""

from __future__ import annotations

import datetime as dt_

import pytest

from dpyr import case_when, col, desc, if_else, lit, n

from conftest import STARWARS


def rows(frame):
    return frame.collect().to_dicts()


# -- filter / mutate ---------------------------------------------------------

def test_filter_drops_na_like_dplyr_S12(make):
    out = rows(make(STARWARS).filter(col.mass > 50))
    assert [r["name"] for r in out] == ["Luke", "Han", "Chewie"]


def test_filter_multiple_predicates(make):
    out = rows(make(STARWARS).filter(col.height > 100, col.mass < 100))
    assert [r["name"] for r in out] == ["Luke", "Leia", "Han"]


def test_mutate_int_division_promotes_S4(make):
    out = rows(make({"x": [1, 2]}).mutate(half=col.x / 2))
    assert out == [{"x": 1, "half": 0.5}, {"x": 2, "half": 1.0}]


def test_mutate_division_by_zero_is_inf_S14(make):
    out = rows(make({"x": [1.0, -1.0, 0.0]}).mutate(y=col.x / 0))
    assert out[0]["y"] == float("inf")
    assert out[1]["y"] == float("-inf")
    import math
    assert math.isnan(out[2]["y"])


def test_mutate_sequential_references(make):
    out = rows(make({"x": [2]}).mutate(y=col.x * 2, z=col.y + 1))
    assert out == [{"x": 2, "y": 4, "z": 5}]


def test_mutate_if_else_and_case_when(make):
    f = make(STARWARS).mutate(
        size=case_when((col.height > 200, "big"), (col.height > 100, "mid"),
                       default="small"),
        heavy=if_else(col.mass > 80, True, False),
    )
    out = rows(f)
    assert [r["size"] for r in out] == ["mid", "mid", "mid", "big", "small", "small"]
    assert out[3]["heavy"] is True and out[1]["heavy"] is False


def test_grouped_mutate_broadcasts_window(make):
    f = (make(STARWARS).group_by(col.species)
         .mutate(rel=col.height - col.height.mean()))
    out = {r["name"]: r["rel"] for r in rows(f.ungroup().arrange(col.name))}
    human_mean = (172 + 150 + 180) / 3
    assert out["Luke"] == pytest.approx(172 - human_mean)
    assert out["Chewie"] == 0.0


def test_grouped_filter_window(make):
    f = (make(STARWARS).group_by(col.species)
         .filter(col.height == col.height.max()))
    names = sorted(r["name"] for r in rows(f.ungroup()))
    assert names == ["Chewie", "Han", "R2D2", "Yoda"]


# -- arrange -----------------------------------------------------------------

def test_arrange_nulls_last_stable_S3(make):
    out = rows(make(STARWARS).arrange(col.mass))
    assert [r["name"] for r in out] == ["Yoda", "Leia", "Luke", "Han", "Chewie", "R2D2"]
    out2 = rows(make(STARWARS).arrange(desc(col.mass)))
    assert [r["name"] for r in out2] == ["Chewie", "Han", "Luke", "Leia", "Yoda", "R2D2"]


def test_arrange_stability_on_ties(make):
    data = {"k": [2, 1, 2, 1], "tag": ["a", "b", "c", "d"]}
    out = rows(make(data).arrange(col.k))
    assert [r["tag"] for r in out] == ["b", "d", "a", "c"]


def test_arrange_by_expression(make):
    out = rows(make({"x": [-3, 1, -2]}).arrange((col.x * col.x)))
    assert [r["x"] for r in out] == [1, -2, -3]


# -- summarize ---------------------------------------------------------------

def test_summarize_grouped_sorted_by_keys_S7(make):
    out = rows(make(STARWARS).group_by(col.species).summarize(n=n()))
    assert [r["species"] for r in out] == ["Droid", "Human", "Wookiee", None]
    assert [r["n"] for r in out] == [1, 3, 1, 1]


def test_summarize_mean_ignores_nulls_S2(make):
    out = rows(make(STARWARS).summarize(m=col.mass.mean()))
    assert out[0]["m"] == pytest.approx((77 + 49 + 80 + 112 + 17) / 5)


def test_summarize_na_rm_false_S2(make):
    out = rows(make(STARWARS).summarize(m=col.mass.mean(na_rm=False)))
    assert out[0]["m"] is None


def test_summarize_sum_all_null_is_zero(make):
    data = {"g": [1, 1], "x": [None, None]}
    f = (make(data).mutate(x=if_else(col.g > 99, 1.0, None))
         .group_by(col.g).summarize(s=col.x.sum()))
    assert rows(f)[0]["s"] == 0.0  # R: sum(c(), na.rm=TRUE) is 0


def test_summarize_n_unique_counts_null(make):
    out = rows(make(STARWARS).summarize(s=col.species.n_unique()))
    assert out[0]["s"] == 4  # Human, Wookiee, Droid, NULL


def test_summarize_bool_sum_counts(make):
    out = rows(make(STARWARS).summarize(k=(col.height > 150).sum()))
    assert out[0]["k"] == 3  # Luke 172, Han 180, Chewie 228


def test_summarize_std_is_sample_std(make):
    out = rows(make({"x": [1.0, 2.0, 3.0]}).summarize(s=col.x.std()))
    assert out[0]["s"] == pytest.approx(1.0)


def test_count_sugar(make):
    out = rows(make(STARWARS).count(col.species))
    assert out == [
        {"species": "Droid", "n": 1}, {"species": "Human", "n": 3},
        {"species": "Wookiee", "n": 1}, {"species": None, "n": 1}]


# -- distinct / slices --------------------------------------------------------

def test_distinct_subset_keeps_first(make):
    out = rows(make(STARWARS).distinct(col.species))
    assert [r["species"] for r in out] == ["Human", "Wookiee", "Droid", None]


def test_slice_head_tail(make):
    f = make(STARWARS).arrange(col.height)
    assert [r["name"] for r in rows(f.slice_head(2))] == ["Yoda", "R2D2"]
    assert [r["name"] for r in rows(f.slice_tail(2))] == ["Han", "Chewie"]


def test_grouped_slice_head(make):
    f = make(STARWARS).arrange(col.name).group_by(col.species).slice_head(1)
    names = sorted(r["name"] for r in rows(f.ungroup()))
    assert names == ["Chewie", "Han", "R2D2", "Yoda"]


def test_slice_sample_is_seeded_subset(make):
    f = make(STARWARS).slice_sample(3, seed=42)
    got = rows(f)
    assert len(got) == 3
    assert {r["name"] for r in got} <= set(STARWARS["name"])


# -- joins --------------------------------------------------------------------

FILMS = {"name": ["Luke", "Leia", "Jabba"], "film": ["ANH", "ANH", "ROTJ"]}


def test_inner_left_join(make):
    sw, films = make(STARWARS), make(FILMS)
    inner = rows(sw.inner_join(films, on=col.name).arrange(col.name))
    assert [r["name"] for r in inner] == ["Leia", "Luke"]
    left = rows(sw.left_join(films, on=col.name).arrange(col.name))
    assert len(left) == 6
    by_name = {r["name"]: r["film"] for r in left}
    assert by_name["Luke"] == "ANH" and by_name["Han"] is None


def test_right_and_full_join(make):
    sw, films = make(STARWARS), make(FILMS)
    right = rows(sw.right_join(films, on=col.name).arrange(col.name))
    assert [r["name"] for r in right] == ["Jabba", "Leia", "Luke"]
    full = rows(sw.full_join(films, on=col.name).arrange(col.name))
    assert len(full) == 7


def test_semi_anti_join(make):
    sw, films = make(STARWARS), make(FILMS)
    assert [r["name"] for r in rows(sw.semi_join(films, on=col.name).arrange(col.name))] \
        == ["Leia", "Luke"]
    assert [r["name"] for r in rows(sw.anti_join(films, on=col.name).arrange(col.name))] \
        == ["Chewie", "Han", "R2D2", "Yoda"]


def test_join_suffixes_S11(make):
    a = make({"k": [1, 2], "v": [10, 20]})
    b = make({"k": [1, 2], "v": [99, 98]})
    out = rows(a.inner_join(b, on=col.k).arrange(col.k))
    assert out == [{"k": 1, "v.x": 10, "v.y": 99}, {"k": 2, "v.x": 20, "v.y": 98}]


def test_join_na_matches_S10(make):
    a = make({"k": ["x", None], "va": [1, 2]})
    b = make({"k": ["x", None], "vb": [3, 4]})
    default = rows(a.inner_join(b, on=col.k).arrange(col.va))
    assert len(default) == 2  # NA matches NA, like dplyr
    never = rows(a.inner_join(b, on=col.k, na_matches="never"))
    assert len(never) == 1


# -- strings, dates, misc expressions ------------------------------------------

def test_string_functions(make):
    f = make(STARWARS).mutate(
        up=col.name.str_to_upper(),
        has_e=col.name.str_detect("e"),
        ln=col.name.str_len(),
        fixed=col.name.str_replace("a", "@"),
    )
    out = rows(f)[0]
    assert out["up"] == "LUKE" and out["has_e"] is True and out["ln"] == 4
    by = {r["name"]: r["fixed"] for r in rows(f)}
    assert by["Han"] == "H@n" and by["Chewie"] == "Chewie"


def test_temporal_functions(make):
    data = {"d": [dt_.date(1977, 5, 25), dt_.date(1980, 6, 20)]}
    out = rows(make(data).mutate(y=col.d.year(), m=col.d.month(), dd=col.d.day()))
    assert out[0] == {"d": dt_.date(1977, 5, 25), "y": 1977, "m": 5, "dd": 25}


def test_is_na_catches_nan_S1(make):
    out = rows(make({"x": [1.0, None, float("nan")]}).mutate(na=col.x.is_na()))
    assert [r["na"] for r in out] == [False, True, True]


def test_is_in_and_between(make):
    out = rows(make(STARWARS).filter(col.name.is_in(["Luke", "Yoda"])))
    assert [r["name"] for r in out] == ["Luke", "Yoda"]
    out2 = rows(make(STARWARS).filter(col.height.between(100, 180)))
    assert [r["name"] for r in out2] == ["Luke", "Leia", "Han"]


def test_floor_mod_matches_python(make):
    out = rows(make({"x": [7, -7]}).mutate(m=col.x % 3))
    assert [r["m"] for r in out] == [1, 2]


def test_pow_and_math(make):
    out = rows(make({"x": [4.0]}).mutate(
        p=col.x ** 2, s=col.x.sqrt(), f=(col.x + 0.5).floor(),
        c=(col.x + 0.5).ceiling(), a=(0 - col.x).abs(), r=lit(2.345).round(1)))
    assert out[0] == {"x": 4.0, "p": 16.0, "s": 2.0, "f": 4, "c": 5,
                      "a": 4.0, "r": 2.3}


# -- reshaping ------------------------------------------------------------------

def test_pivot_longer(make):
    data = {"id": ["a", "b"], "x": [1.0, 2.0], "y": [3.0, 4.0]}
    out = rows(make(data).pivot_longer([col.x, col.y]).arrange(col.id, col.name))
    assert out == [
        {"id": "a", "name": "x", "value": 1.0},
        {"id": "a", "name": "y", "value": 3.0},
        {"id": "b", "name": "x", "value": 2.0},
        {"id": "b", "name": "y", "value": 4.0},
    ]


def test_pivot_longer_keeps_nulls(make):
    data = {"id": ["a"], "x": [1.0], "y": [None]}
    out = rows(make(data).pivot_longer([col.x, col.y]).arrange(col.name))
    assert out[1]["value"] is None


def test_pivot_wider_roundtrip(make):
    data = {"id": ["a", "a", "b", "b"], "key": ["x", "y", "x", "y"],
            "val": [1.0, 2.0, 3.0, 4.0]}
    wide = make(data).pivot_wider(names_from=col.key, values_from=col.val)
    assert wide.columns == ["id", "x", "y"]
    out = rows(wide.arrange(col.id))
    assert out == [{"id": "a", "x": 1.0, "y": 2.0}, {"id": "b", "x": 3.0, "y": 4.0}]


def test_pull_and_getitem(make):
    f = make(STARWARS)
    assert f.pull(col.name)[:2] == ["Luke", "Leia"]
    assert f["height"][0] == 172
