import pytest

from dpyr import (
    BOOL,
    DATE,
    FLOAT64,
    INT64,
    STR,
    BoolExpr,
    ExprTypeError,
    NumExpr,
    StrExpr,
    TemporalExpr,
    infer_dtype,
    typed_col,
)


def test_factory_picks_class_by_dtype():
    assert isinstance(typed_col("h", INT64), NumExpr)
    assert isinstance(typed_col("h", FLOAT64), NumExpr)
    assert isinstance(typed_col("s", STR), StrExpr)
    assert isinstance(typed_col("b", BOOL), BoolExpr)
    assert isinstance(typed_col("d", DATE), TemporalExpr)


def test_typed_cols_are_cols_in_the_ir():
    e = typed_col("height", INT64) > 180
    assert infer_dtype(e, {"height": INT64}) == BOOL
    assert repr(typed_col("height", INT64)) == "col.height"


def test_inapplicable_method_fails_at_build_time():
    with pytest.raises(ExprTypeError, match="not available on a StrExpr"):
        typed_col("name", STR).mean()
    with pytest.raises(ExprTypeError, match="not available on a NumExpr"):
        typed_col("height", INT64).str_detect("x")
    with pytest.raises(ExprTypeError, match="not available on a TemporalExpr"):
        typed_col("born", DATE).sqrt()


def test_applicable_methods_still_work():
    assert infer_dtype(typed_col("h", INT64).mean(), {"h": INT64}) == FLOAT64
    assert infer_dtype(typed_col("s", STR).str_len(), {"s": STR}) == INT64
    assert infer_dtype(typed_col("b", BOOL).sum(), {"b": BOOL}) == INT64
    assert infer_dtype(typed_col("d", DATE).year(), {"d": DATE}) == INT64
