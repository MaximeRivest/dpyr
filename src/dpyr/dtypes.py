"""Dtype system (ROADMAP 1.1).

Pinned decisions from SEMANTICS.md:
- S1: missing values are typed nulls; NULL is the dtype of an all-null literal.
- S4: int / int promotes to float, like R.
- S13: counts are INT64.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    name: str

    def __repr__(self) -> str:
        return self.name


INT64 = DType("Int64")
FLOAT64 = DType("Float64")
BOOL = DType("Bool")
STR = DType("Str")
DATE = DType("Date")
DATETIME = DType("Datetime")
NULL = DType("Null")

ALL_DTYPES = (INT64, FLOAT64, BOOL, STR, DATE, DATETIME, NULL)
NUMERIC = (INT64, FLOAT64)
TEMPORAL = (DATE, DATETIME)


def is_numeric(dt: DType) -> bool:
    return dt in NUMERIC


def unify(a: DType, b: DType) -> DType | None:
    """Common supertype for branch results (if_else, case_when, fill values).

    NULL unifies with anything (S1); INT64 widens to FLOAT64. Anything else
    must match exactly. Returns None when no unification exists.
    """
    if a == b:
        return a
    if a == NULL:
        return b
    if b == NULL:
        return a
    if {a, b} == {INT64, FLOAT64}:
        return FLOAT64
    return None


def arith_result(op: str, a: DType, b: DType) -> DType | None:
    """Result dtype of a binary arithmetic op, or None if invalid."""
    if a == NULL or b == NULL:
        a = a if a != NULL else (b if b != NULL else INT64)
        b = a if b == NULL else b
    if not (is_numeric(a) and is_numeric(b)):
        # str + str concatenation is deliberately NOT supported (use str_c later)
        return None
    if op == "/":
        return FLOAT64  # S4: int / int -> float, like R
    if op == "//":
        return INT64 if (a, b) == (INT64, INT64) else FLOAT64
    return FLOAT64 if FLOAT64 in (a, b) else INT64
