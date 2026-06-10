import datetime as dt_

import pytest

from dpyr import (
    BOOL,
    DATE,
    FLOAT64,
    INT64,
    STR,
    ColumnNotFoundError,
    ExprTypeError,
    case_when,
    col,
    desc,
    if_else,
    infer_dtype,
    n,
)

SCHEMA = {"height": INT64, "mass": FLOAT64, "name": STR, "alive": BOOL,
          "born": DATE}


def t(e):
    return infer_dtype(e, SCHEMA)


def test_column_lookup():
    assert t(col.height) == INT64
    assert t(col.name) == STR


def test_unknown_column_raises_immediately_with_suggestion():
    with pytest.raises(ColumnNotFoundError, match="Did you mean 'height'"):
        t(col.heigth > 180)


def test_comparison_returns_bool():
    assert t(col.height > 180) == BOOL
    assert t(col.name == "Luke") == BOOL


def test_arithmetic_inference():
    assert t(col.height + 1) == INT64
    assert t(col.height / 100) == FLOAT64  # S4
    assert t(col.mass * col.height) == FLOAT64
    assert t((col.mass / (col.height / 100) ** 2)) == FLOAT64


def test_boolean_ops():
    assert t((col.height > 180) & col.alive) == BOOL
    assert t(~col.alive) == BOOL


def test_boolean_on_non_bool_raises():
    with pytest.raises(ExprTypeError):
        t(col.height & col.alive)
    with pytest.raises(ExprTypeError):
        t(~col.name)


def test_arith_on_string_raises():
    with pytest.raises(ExprTypeError):
        t(col.name + 1)


def test_compare_incompatible_raises():
    with pytest.raises(ExprTypeError):
        t(col.name > col.height)


def test_python_bool_coercion_is_a_helpful_error():
    with pytest.raises(ExprTypeError, match="& | ~"):
        bool(col.alive)


def test_aggregations():
    assert t(col.mass.mean()) == FLOAT64
    assert t(col.height.sum()) == INT64
    assert t(col.height.min()) == INT64
    assert t(col.name.n_unique()) == INT64
    assert t(n()) == INT64
    assert t(col.alive.sum()) == INT64  # count of trues


def test_mean_of_string_raises():
    with pytest.raises(ExprTypeError, match="numeric"):
        t(col.name.mean())


def test_nested_aggregation_raises():
    with pytest.raises(ExprTypeError, match="nested"):
        t(col.mass.mean().sum())


def test_string_functions():
    assert t(col.name.str_detect("Sky")) == BOOL
    assert t(col.name.str_len()) == INT64
    assert t(col.name.str_to_lower()) == STR
    with pytest.raises(ExprTypeError):
        t(col.height.str_detect("x"))


def test_temporal_functions():
    assert t(col.born.year()) == INT64
    with pytest.raises(ExprTypeError):
        t(col.name.year())


def test_is_na_and_is_in():
    assert t(col.mass.is_na()) == BOOL
    assert t(col.name.is_in(["Luke", "Leia"])) == BOOL


def test_if_else():
    assert t(if_else(col.alive, 1, 0)) == INT64
    assert t(if_else(col.alive, 1, 0.5)) == FLOAT64
    assert t(if_else(col.alive, "y", None)) == STR
    with pytest.raises(ExprTypeError, match="condition"):
        t(if_else(col.height, 1, 0))
    with pytest.raises(ExprTypeError, match="incompatible"):
        t(if_else(col.alive, "y", 1))


def test_case_when():
    e = case_when((col.height > 200, "tall"), (col.height > 150, "mid"),
                  default="short")
    assert t(e) == STR
    with pytest.raises(ExprTypeError, match="incompatible"):
        t(case_when((col.alive, 1), (col.height > 0, "x")))
    with pytest.raises(ExprTypeError):
        case_when()


def test_desc_only_valid_in_arrange():
    with pytest.raises(ExprTypeError, match="arrange"):
        t(desc(col.height))


def test_literal_dtypes():
    assert t(col.height > 1.5) == BOOL
    assert t(col.born == dt_.date(1977, 5, 25)) == BOOL


def test_repr_is_stable_and_canonical():
    e = (col.mass / (col.height / 100) ** 2).mean(na_rm=False)
    assert repr(e) == ("(col.mass / pow((col.height / lit(100)), lit(2)))"
                       ".mean(na_rm=False)")
