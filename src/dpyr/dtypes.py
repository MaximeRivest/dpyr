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
    nested: bool = False  # List/Array/Struct: carried through, not computed on (S35)

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


def nested(name: str) -> DType:
    """A List/Array/Struct dtype, named canonically (e.g. 'List(Str)'), so
    the same data has the same dtype on both engines."""
    return DType(name, nested=True)


def is_nested(dt: DType) -> bool:
    return dt.nested


def _top_level_split(body: str) -> list[str]:
    """Split 'a: Str, b: List(Int64)' at the commas outside parentheses."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(body[start:i].strip())
            start = i + 1
    parts.append(body[start:].strip())
    return parts


def from_name(name: str) -> DType:
    """The dtype a canonical name stands for ('Str', 'List(Int64)', ...)."""
    for d in ALL_DTYPES:
        if d.name == name:
            return d
    return nested(name)


def list_inner(d: DType) -> DType | None:
    """The element dtype of a List or Array dtype; None for anything else."""
    if d.name.startswith("List(") and d.name.endswith(")"):
        return from_name(d.name[len("List("):-1])
    if d.name.startswith("Array(") and d.name.endswith(")"):
        inner, _size = d.name[len("Array("):-1].rsplit(",", 1)
        return from_name(inner.strip())
    return None


def struct_fields(d: DType) -> list[tuple[str, DType]] | None:
    """The (name, dtype) fields of a Struct dtype, in order; None for
    anything else. Read back from the canonical name, so a field name that
    itself contains ', ' or ': ' or parentheses can't be read (an error)."""
    if not (d.name.startswith("Struct(") and d.name.endswith(")")):
        return None
    body = d.name[len("Struct("):-1]
    if not body:
        return []
    fields = []
    for part in _top_level_split(body):
        name, sep, type_name = part.partition(": ")
        known = (any(type_name == x.name for x in ALL_DTYPES)
                 or type_name.startswith(("List(", "Array(", "Struct(")))
        if not sep or not name or not known:
            raise ValueError(f"can't read the fields of {d.name}")
        fields.append((name, from_name(type_name)))
    return fields


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
