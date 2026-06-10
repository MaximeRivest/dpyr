"""Logical plan nodes with synchronous schema inference (ROADMAP 1.4, 1.5).

Every node validates its inputs and computes `schema` (ordered dict of
column -> dtype) and `groups` (active group keys) at construction time.
A bad verb call therefore raises on the line that made it.

Node reprs are canonical; `plan_hash` derives the materialization-cache key
from them (DESIGN.md §3).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

from . import dtypes as dt
from .dtypes import DType
from .errors import (
    ColumnNotFoundError,
    DuplicateColumnError,
    ExprTypeError,
    GroupError,
)
from .expr import Agg, Col, Desc, Expr, N, contains_agg, infer_dtype

Schema = dict[str, DType]
JoinHow = Literal["inner", "left", "right", "full", "semi", "anti"]


@dataclass(frozen=True)
class PlanNode:
    schema: Schema = field(init=False)
    groups: tuple[str, ...] = field(init=False)

    def _finish(self, schema: Schema, groups: tuple[str, ...]) -> None:
        object.__setattr__(self, "schema", dict(schema))
        object.__setattr__(self, "groups", groups)

    def __repr__(self) -> str:  # overridden by every subclass
        raise NotImplementedError


def _check_cols(names: list[str], schema: Schema, context: str) -> None:
    for name in names:
        if name not in schema:
            raise ColumnNotFoundError(name, schema, context)


@dataclass(frozen=True, repr=False)
class Source(PlanNode):
    """A named table with a known schema (file, in-memory frame, db table).

    `token` identifies the data payload in the backend registry and
    participates in the repr, so plan hashes distinguish different data.
    """
    name: str
    source_schema: tuple[tuple[str, DType], ...]
    token: str = ""

    def __post_init__(self) -> None:
        self._finish(dict(self.source_schema), ())

    def __repr__(self) -> str:
        cols = ", ".join(f"{k}: {v!r}" for k, v in self.source_schema)
        return f"source({self.name!r}, {self.token!r}, {{{cols}}})"


@dataclass(frozen=True, repr=False)
class Filter(PlanNode):
    child: PlanNode
    predicates: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if not self.predicates:
            raise ExprTypeError("filter() needs at least one condition")
        for p in self.predicates:
            out = infer_dtype(p, self.child.schema, context="filter()")
            if out not in (dt.BOOL, dt.NULL):
                raise ExprTypeError(f"filter() condition must be boolean, got {out!r}: {p!r}")
        self._finish(self.child.schema, self.child.groups)

    def __repr__(self) -> str:
        return f"{self.child!r}.filter({', '.join(map(repr, self.predicates))})"


@dataclass(frozen=True, repr=False)
class Mutate(PlanNode):
    child: PlanNode
    exprs: tuple[tuple[str, Expr], ...]

    def __post_init__(self) -> None:
        # dplyr semantics: columns are created/overwritten left to right and
        # later expressions can reference earlier ones.
        schema = dict(self.child.schema)
        for name, e in self.exprs:
            schema[name] = infer_dtype(e, schema, context=f"mutate({name}=...)")
        self._finish(schema, self.child.groups)

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.exprs)
        return f"{self.child!r}.mutate({body})"


@dataclass(frozen=True, repr=False)
class Select(PlanNode):
    child: PlanNode
    keep: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_cols(list(self.keep), self.child.schema, "select()")
        if len(set(self.keep)) != len(self.keep):
            dup = next(k for k in self.keep if self.keep.count(k) > 1)
            raise DuplicateColumnError(dup, "select()")
        # dplyr keeps group columns even when not selected
        keep = list(self.keep)
        for g in self.child.groups:
            if g not in keep:
                keep.insert(0, g)
        self._finish({k: self.child.schema[k] for k in keep}, self.child.groups)

    def __repr__(self) -> str:
        return f"{self.child!r}.select({', '.join(self.keep)})"


@dataclass(frozen=True, repr=False)
class Rename(PlanNode):
    child: PlanNode
    mapping: tuple[tuple[str, str], ...]  # (new, old) pairs, dplyr-style

    def __post_init__(self) -> None:
        olds = [old for _, old in self.mapping]
        _check_cols(olds, self.child.schema, "rename()")
        lookup = {old: new for new, old in self.mapping}
        schema: Schema = {}
        for k, v in self.child.schema.items():
            new = lookup.get(k, k)
            if new in schema:
                raise DuplicateColumnError(new, "rename()")
            schema[new] = v
        groups = tuple(lookup.get(g, g) for g in self.child.groups)
        self._finish(schema, groups)

    def __repr__(self) -> str:
        body = ", ".join(f"{new}={old}" for new, old in self.mapping)
        return f"{self.child!r}.rename({body})"


@dataclass(frozen=True, repr=False)
class Arrange(PlanNode):
    child: PlanNode
    keys: tuple[Expr, ...]  # Expr or Desc(expr); S3 pins stability & NA-last

    def __post_init__(self) -> None:
        if not self.keys:
            raise ExprTypeError("arrange() needs at least one sort key")
        for k in self.keys:
            inner = k.operand if isinstance(k, Desc) else k
            infer_dtype(inner, self.child.schema, context="arrange()")
        self._finish(self.child.schema, self.child.groups)

    def __repr__(self) -> str:
        return f"{self.child!r}.arrange({', '.join(map(repr, self.keys))})"


@dataclass(frozen=True, repr=False)
class Distinct(PlanNode):
    child: PlanNode
    cols: tuple[str, ...]  # empty = all columns

    def __post_init__(self) -> None:
        _check_cols(list(self.cols), self.child.schema, "distinct()")
        if self.cols:  # dplyr: distinct(df, a, b) keeps only those (+ groups)
            keep = list(dict.fromkeys((*self.child.groups, *self.cols)))
            schema = {k: self.child.schema[k] for k in keep}
        else:
            schema = self.child.schema
        self._finish(schema, self.child.groups)

    def __repr__(self) -> str:
        return f"{self.child!r}.distinct({', '.join(self.cols)})"


@dataclass(frozen=True, repr=False)
class Slice(PlanNode):
    child: PlanNode
    kind: Literal["head", "tail", "sample"]
    n: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ExprTypeError(f"slice_{self.kind}() needs n >= 0, got {self.n}")
        self._finish(self.child.schema, self.child.groups)

    def __repr__(self) -> str:
        seed = f", seed={self.seed}" if self.seed is not None else ""
        return f"{self.child!r}.slice_{self.kind}({self.n}{seed})"


@dataclass(frozen=True, repr=False)
class GroupBy(PlanNode):
    child: PlanNode
    keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.keys:
            raise GroupError("group_by() needs at least one column")
        _check_cols(list(self.keys), self.child.schema, "group_by()")
        self._finish(self.child.schema, self.keys)

    def __repr__(self) -> str:
        return f"{self.child!r}.group_by({', '.join(self.keys)})"


@dataclass(frozen=True, repr=False)
class Summarize(PlanNode):
    child: PlanNode
    aggs: tuple[tuple[str, Expr], ...]

    def __post_init__(self) -> None:
        if not self.aggs:
            raise ExprTypeError("summarize() needs at least one aggregation")
        # S7: output ordered group keys first, sorted result. S9: drops the
        # last grouping level.
        schema: Schema = {g: self.child.schema[g] for g in self.child.groups}
        for name, e in self.aggs:
            if not contains_agg(e):
                raise ExprTypeError(
                    f"summarize({name}=...) must aggregate (use .mean(), n(), ...): {e!r}")
            if name in schema:
                raise DuplicateColumnError(name, "summarize()")
            schema[name] = infer_dtype(e, self.child.schema, context=f"summarize({name}=...)")
        self._finish(schema, self.child.groups[:-1])

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.aggs)
        return f"{self.child!r}.summarize({body})"


@dataclass(frozen=True, repr=False)
class Join(PlanNode):
    left: PlanNode
    right: PlanNode
    how: JoinHow
    on: tuple[str, ...]
    suffix: tuple[str, str] = (".x", ".y")  # S11
    na_matches: Literal["na", "never"] = "na"  # S10

    def __post_init__(self) -> None:
        _check_cols(list(self.on), self.left.schema, f"{self.how}_join() left")
        _check_cols(list(self.on), self.right.schema, f"{self.how}_join() right")
        for k in self.on:
            if dt.unify(self.left.schema[k], self.right.schema[k]) is None:
                raise ExprTypeError(
                    f"join key '{k}' has incompatible dtypes: "
                    f"{self.left.schema[k]!r} vs {self.right.schema[k]!r}")
        if self.how in ("semi", "anti"):  # filtering joins keep left schema
            self._finish(dict(self.left.schema), self.left.groups)
            return
        overlap = (set(self.left.schema) & set(self.right.schema)) - set(self.on)
        # dplyr keeps left columns in their original positions, suffixed in place
        schema: Schema = {}
        for name, dtype in self.left.schema.items():
            schema[name + self.suffix[0] if name in overlap else name] = dtype
        for name, dtype in self.right.schema.items():
            if name in self.on:
                continue
            schema[name + self.suffix[1] if name in overlap else name] = dtype
        groups = tuple(g + self.suffix[0] if g in overlap else g for g in self.left.groups)
        self._finish(schema, groups)

    def __repr__(self) -> str:
        return (f"{self.left!r}.{self.how}_join({self.right!r}, "
                f"on=({', '.join(self.on)}))")


@dataclass(frozen=True, repr=False)
class PivotLonger(PlanNode):
    child: PlanNode
    cols: tuple[str, ...]
    names_to: str = "name"
    values_to: str = "value"

    def __post_init__(self) -> None:
        if not self.cols:
            raise ExprTypeError("pivot_longer() needs at least one column")
        _check_cols(list(self.cols), self.child.schema, "pivot_longer()")
        value_type: DType = dt.NULL
        for c in self.cols:
            unified = dt.unify(value_type, self.child.schema[c])
            if unified is None:
                raise ExprTypeError(
                    f"pivot_longer() columns have incompatible dtypes: '{c}' is "
                    f"{self.child.schema[c]!r}, expected {value_type!r}")
            value_type = unified
        schema = {k: v for k, v in self.child.schema.items() if k not in self.cols}
        for new in (self.names_to, self.values_to):
            if new in schema:
                raise DuplicateColumnError(new, "pivot_longer()")
        schema[self.names_to] = dt.STR
        schema[self.values_to] = value_type
        self._finish(schema, self.child.groups)

    def __repr__(self) -> str:
        return (f"{self.child!r}.pivot_longer(({', '.join(self.cols)}), "
                f"names_to={self.names_to!r}, values_to={self.values_to!r})")


@dataclass(frozen=True, repr=False)
class PivotWider(PlanNode):
    """Output columns come from data values: schema needs data (DESIGN §3).

    Validation of inputs is eager; the output schema is a placeholder until
    materialization implicitly persists the input (ROADMAP 3.4 / 6.3).
    """
    child: PlanNode
    names_from: str
    values_from: str

    def __post_init__(self) -> None:
        _check_cols([self.names_from, self.values_from], self.child.schema, "pivot_wider()")
        schema = {k: v for k, v in self.child.schema.items()
                  if k not in (self.names_from, self.values_from)}
        self._finish(schema, self.child.groups)

    @property
    def schema_requires_data(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (f"{self.child!r}.pivot_wider(names_from={self.names_from!r}, "
                f"values_from={self.values_from!r})")


def plan_hash(node: PlanNode) -> str:
    """Stable cache key for the materialization cache (DESIGN §3)."""
    return hashlib.sha256(repr(node).encode()).hexdigest()[:16]


def _used_agg_inputs(e: Expr) -> set[str]:
    cols: set[str] = set()

    def rec(x: Expr) -> None:
        if isinstance(x, Col):
            cols.add(x.name)
        for f in getattr(x, "__dataclass_fields__", {}):
            v = getattr(x, f)
            if isinstance(v, Expr):
                rec(v)
            elif isinstance(v, tuple):
                for item in v:
                    if isinstance(item, Expr):
                        rec(item)
                    elif isinstance(item, tuple):
                        for sub in item:
                            if isinstance(sub, Expr):
                                rec(sub)

    rec(e)
    return cols


def used_columns(e: Expr) -> set[str]:
    """All column names referenced by an expression (for pushdown later)."""
    return _used_agg_inputs(e)


__all__ = [
    "PlanNode", "Source", "Filter", "Mutate", "Select", "Rename", "Arrange",
    "Distinct", "Slice", "GroupBy", "Summarize", "Join", "PivotLonger",
    "PivotWider", "plan_hash", "used_columns", "Schema", "JoinHow",
    "Agg", "Col", "Desc", "Expr", "N",
]
