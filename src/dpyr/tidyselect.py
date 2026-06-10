"""tidyselect (Epic 6): starts_with, ends_with, contains, matches, where,
everything, negation — resolved against a schema at verb-call time.

Selection semantics follow tidyselect: positive selectors expand in schema
order (first match wins the position); a selection consisting only of
negations means "everything except".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .dtypes import DType
from .errors import ColumnNotFoundError, ExprTypeError
from .expr import Col, UnaryOp

Schema = dict[str, DType]


@dataclass(frozen=True)
class Selector:
    predicate: Callable[[str, DType], bool]
    label: str
    negate: bool = False

    def __neg__(self) -> Selector:
        return Selector(self.predicate, f"-{self.label}", not self.negate)

    def matches(self, schema: Schema) -> list[str]:
        return [k for k, v in schema.items() if self.predicate(k, v)]


def starts_with(prefix: str) -> Selector:
    return Selector(lambda k, v: k.startswith(prefix), f"starts_with({prefix!r})")


def ends_with(suffix: str) -> Selector:
    return Selector(lambda k, v: k.endswith(suffix), f"ends_with({suffix!r})")


def contains(s: str) -> Selector:
    return Selector(lambda k, v: s in k, f"contains({s!r})")


def matches(pattern: str) -> Selector:
    rx = re.compile(pattern)
    return Selector(lambda k, v: rx.search(k) is not None, f"matches({pattern!r})")


def where(pred: Callable[[DType], bool]) -> Selector:
    return Selector(lambda k, v: bool(pred(v)), "where(...)")


def everything() -> Selector:
    return Selector(lambda k, v: True, "everything()")


def is_numeric(d: DType) -> bool:
    from . import dtypes as dt
    return dt.is_numeric(d)


def is_string(d: DType) -> bool:
    from . import dtypes as dt
    return d == dt.STR


def is_bool(d: DType) -> bool:
    from . import dtypes as dt
    return d == dt.BOOL


def resolve_selection(items: tuple[Any, ...], schema: Schema,
                      context: str) -> tuple[str, ...]:
    """Expand a mix of names, col refs, selectors and negations into an
    ordered tuple of column names."""
    keep: list[str] = []
    drop: set[str] = set()
    saw_positive = False

    def add(name: str) -> None:
        if name not in keep:
            keep.append(name)

    for item in items:
        if isinstance(item, str):
            if item.startswith("-") and item[1:] in schema:
                drop.add(item[1:])
                continue
            if item not in schema:
                raise ColumnNotFoundError(item, schema, context)
            saw_positive = True
            add(item)
        elif isinstance(item, Col):
            if item.name not in schema:
                raise ColumnNotFoundError(item.name, schema, context)
            saw_positive = True
            add(item.name)
        elif isinstance(item, UnaryOp) and item.op == "-" and isinstance(item.operand, Col):
            if item.operand.name not in schema:
                raise ColumnNotFoundError(item.operand.name, schema, context)
            drop.add(item.operand.name)
        elif isinstance(item, Selector):
            matched = item.matches(schema)
            if item.negate:
                drop.update(matched)
            else:
                saw_positive = True
                for m in matched:
                    add(m)
        else:
            raise ExprTypeError(
                f"{context} can't interpret {item!r}; expected a column name, "
                "col.<name>, a selector (starts_with, where, ...) or a negation")

    if not saw_positive:  # pure negation: everything except
        keep = list(schema)
    return tuple(k for k in keep if k not in drop)


_FN_SHORTCUTS: dict[str, Callable[[Col], Any]] = {
    "mean": lambda c: c.mean(), "median": lambda c: c.median(),
    "sum": lambda c: c.sum(), "min": lambda c: c.min(),
    "max": lambda c: c.max(), "std": lambda c: c.std(),
    "var": lambda c: c.var(), "n_unique": lambda c: c.n_unique(),
    "first": lambda c: c.first(), "last": lambda c: c.last(),
}

IntoFn = Callable[[Col], Any] | str


@dataclass(frozen=True)
class Across:
    """across(selector, fns): apply fns to every selected column.
    Expanded against the frame's schema at verb-call time."""

    selector: Selector
    fns: tuple[tuple[str, Callable[[Col], Any]], ...]
    names: str  # template with {col} and {fn}

    def expand(self, schema: Schema) -> list[tuple[str, Any]]:
        from .expr import typed_col
        out = []
        for colname in self.selector.matches(schema):
            for fn_name, fn in self.fns:
                label = self.names.format(col=colname, fn=fn_name)
                out.append((label, fn(typed_col(colname, schema[colname]))))
        return out


def _into_fns(fns: IntoFn | list[IntoFn] | dict[str, IntoFn]
              ) -> tuple[tuple[str, Callable[[Col], Any]], ...]:
    def one(name_hint: str, f: IntoFn) -> tuple[str, Callable[[Col], Any]]:
        if isinstance(f, str):
            if f not in _FN_SHORTCUTS:
                raise ExprTypeError(
                    f"across() unknown function shortcut {f!r}; "
                    f"known: {', '.join(sorted(_FN_SHORTCUTS))}")
            return (f, _FN_SHORTCUTS[f])
        return (name_hint or getattr(f, "__name__", "fn"), f)

    if isinstance(fns, dict):
        return tuple(one(k, v) for k, v in fns.items())
    if isinstance(fns, list):
        return tuple(one(f if isinstance(f, str) else "", f) for f in fns)
    return (one(fns if isinstance(fns, str) else "", fns),)


def across(selector: Selector | Col | str,
           fns: IntoFn | list[IntoFn] | dict[str, IntoFn],
           names: str | None = None) -> Across:
    if isinstance(selector, Col):
        target = selector.name
        sel = Selector(lambda k, v: k == target, f"col({selector.name})")
    elif isinstance(selector, str):
        target = selector
        sel = Selector(lambda k, v: k == target, f"col({selector})")
    else:
        sel = selector
    resolved = _into_fns(fns)
    if names is None:
        names = "{col}" if len(resolved) == 1 else "{col}_{fn}"
    return Across(sel, resolved, names)


__all__ = ["starts_with", "ends_with", "contains", "matches", "where",
           "everything", "is_numeric", "is_string", "is_bool", "Selector",
           "resolve_selection", "across", "Across"]
