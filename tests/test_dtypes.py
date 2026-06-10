from dpyr import BOOL, FLOAT64, INT64, NULL, STR
from dpyr.dtypes import arith_result, is_numeric, unify


def test_unify_identity_and_null():
    assert unify(INT64, INT64) == INT64
    assert unify(NULL, STR) == STR
    assert unify(STR, NULL) == STR
    assert unify(NULL, NULL) == NULL


def test_unify_numeric_widening():
    assert unify(INT64, FLOAT64) == FLOAT64
    assert unify(FLOAT64, INT64) == FLOAT64


def test_unify_incompatible():
    assert unify(STR, INT64) is None
    assert unify(BOOL, FLOAT64) is None


def test_int_division_promotes_to_float_S4():
    assert arith_result("/", INT64, INT64) == FLOAT64


def test_arith_int_preserved():
    assert arith_result("+", INT64, INT64) == INT64
    assert arith_result("*", INT64, FLOAT64) == FLOAT64
    assert arith_result("//", INT64, INT64) == INT64


def test_arith_invalid_types():
    assert arith_result("+", STR, STR) is None
    assert arith_result("-", BOOL, INT64) is None


def test_is_numeric():
    assert is_numeric(INT64) and is_numeric(FLOAT64)
    assert not is_numeric(STR) and not is_numeric(BOOL)
