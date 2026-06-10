import pytest

from dpyr import (
    FLOAT64,
    INT64,
    STR,
    ColumnNotFoundError,
    DuplicateColumnError,
    ExprTypeError,
    GroupedDFrame,
    GroupError,
    col,
    desc,
    from_schema,
    n,
)

sw = from_schema(
    {"name": STR, "height": INT64, "mass": FLOAT64, "species": STR},
    name="starwars",
)


# -- filter --------------------------------------------------------------

def test_filter_keeps_schema():
    out = sw.filter(col.height > 180, col.mass < 100)
    assert out.schema == sw.schema


def test_filter_requires_boolean():
    with pytest.raises(ExprTypeError, match="boolean"):
        sw.filter(col.height + 1)


def test_filter_bad_column_raises_on_this_line():
    with pytest.raises(ColumnNotFoundError, match="Did you mean 'height'"):
        sw.filter(col.heigth > 180)


# -- mutate --------------------------------------------------------------

def test_mutate_adds_and_overwrites():
    out = sw.mutate(bmi=col.mass / (col.height / 100) ** 2, height=col.height + 1)
    assert out.schema["bmi"] == FLOAT64
    assert out.schema["height"] == INT64
    assert out.columns[-1] == "bmi"


def test_mutate_sees_earlier_columns():
    out = sw.mutate(h2=col.height * 2, h4=col.h2 * 2)
    assert out.schema["h4"] == INT64


def test_mutate_cannot_see_later_columns():
    with pytest.raises(ColumnNotFoundError):
        sw.mutate(h4=col.h2 * 2, h2=col.height * 2)


# -- select / rename -------------------------------------------------------

def test_select_subsets_in_order():
    out = sw.select(col.mass, "name")
    assert out.columns == ["mass", "name"]


def test_select_duplicates_dedupe_like_tidyselect():
    assert sw.select(col.name, col.name).columns == ["name"]


def test_select_keeps_group_columns():
    out = sw.group_by(col.species).select(col.height)
    assert out.columns == ["species", "height"]


def test_rename_new_eq_old():
    out = sw.rename(nm=col.name)
    assert "nm" in out.schema and "name" not in out.schema
    assert out.schema["nm"] == STR


def test_rename_collision_raises():
    with pytest.raises(DuplicateColumnError):
        sw.rename(mass=col.height)


# -- arrange / distinct / slice ---------------------------------------------

def test_arrange_with_desc():
    out = sw.arrange(desc(col.mass), col.name)
    assert out.schema == sw.schema


def test_arrange_bad_column():
    with pytest.raises(ColumnNotFoundError):
        sw.arrange(col.weight)


def test_distinct_all_and_subset():
    assert sw.distinct().schema == sw.schema
    assert sw.distinct(col.species).columns == ["species"]


def test_slice_head_negative_raises():
    with pytest.raises(ExprTypeError):
        sw.slice_head(-1)


# -- group_by / summarize ----------------------------------------------------

def test_summarize_grouped_schema_and_ungroup_S7_S9():
    out = sw.group_by(col.species).summarize(n=n(), mh=col.height.mean())
    assert not isinstance(out, GroupedDFrame)  # one level -> fully ungrouped
    assert out.columns == ["species", "n", "mh"]
    assert out.schema == {"species": STR, "n": INT64, "mh": FLOAT64}


def test_summarize_drops_last_group_level_only_S9():
    g2 = sw.group_by(col.species, col.name).summarize(mh=col.height.mean())
    assert isinstance(g2, GroupedDFrame)
    assert g2.groups == ("species",)


def test_summarize_requires_aggregate():
    with pytest.raises(ExprTypeError, match="must aggregate"):
        sw.group_by(col.species).summarize(h=col.height + 1)


def test_summarize_name_collision_with_group_key():
    with pytest.raises(DuplicateColumnError):
        sw.group_by(col.species).summarize(species=n())


def test_ungroup():
    g = sw.group_by(col.species)
    assert g.ungroup().schema == sw.schema


def test_grouped_mutate_stays_grouped():
    g = sw.group_by(col.species).mutate(rel=col.height - col.height.mean())
    assert isinstance(g, GroupedDFrame)
    assert g.schema["rel"] == FLOAT64


def test_group_by_unknown_column():
    with pytest.raises(ColumnNotFoundError):
        sw.group_by(col.specis)


def test_group_by_no_keys():
    with pytest.raises(GroupError):
        sw.group_by()


def test_count_sugar():
    out = sw.count(col.species)
    assert out.schema == {"species": STR, "n": INT64}
    assert sw.count().schema == {"n": INT64}


# -- joins ---------------------------------------------------------------

films = from_schema({"name": STR, "film": STR, "mass": INT64}, name="films")


def test_left_join_suffixes_S11():
    out = sw.left_join(films, on=col.name)
    assert "mass.x" in out.schema and "mass.y" in out.schema
    assert out.schema["mass.x"] == FLOAT64 and out.schema["mass.y"] == INT64
    assert "film" in out.schema


def test_semi_join_keeps_left_schema():
    out = sw.semi_join(films, on="name")
    assert out.schema == sw.schema


def test_join_key_missing():
    with pytest.raises(ColumnNotFoundError):
        sw.inner_join(films, on=col.film)


def test_join_key_type_mismatch():
    other = from_schema({"name": INT64}, name="ids")
    with pytest.raises(ExprTypeError, match="incompatible"):
        sw.inner_join(other, on=col.name)


# -- pivots ---------------------------------------------------------------

def test_pivot_longer_schema():
    out = sw.pivot_longer([col.height, col.mass], names_to="dim", values_to="val")
    assert out.schema == {"name": STR, "species": STR, "dim": STR, "val": FLOAT64}


def test_pivot_longer_incompatible_dtypes():
    with pytest.raises(ExprTypeError, match="incompatible"):
        sw.pivot_longer([col.name, col.height])


def test_pivot_wider_validates_eagerly():
    out = sw.pivot_wider(names_from=col.species, values_from=col.mass)
    assert out.plan.schema_requires_data
    with pytest.raises(ColumnNotFoundError):
        sw.pivot_wider(names_from=col.speciez, values_from=col.mass)


# -- ergonomics -------------------------------------------------------------

def test_dir_exposes_columns_for_completion():
    assert "height" in dir(sw)
