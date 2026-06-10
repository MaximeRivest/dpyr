"""Expression IR (ROADMAP 1.2, 1.3).

Expressions are immutable trees built by the `col` proxy and helper
functions. They know nothing about data; `infer_dtype(expr, schema)` is the
single place where an expression meets a schema, performing validation and
type inference (the schema-eager half of DESIGN.md §3).

Every node's repr is stable and canonical — plan hashing (ROADMAP 1.5) and
IR snapshot tests depend on it.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Union

from . import dtypes as dt
from .dtypes import DType
from .errors import ColumnNotFoundError, ExprTypeError

Schema = dict[str, DType]

_COMPARISONS = {"==", "!=", "<", "<=", ">", ">="}
_ARITHMETIC = {"+", "-", "*", "/", "//", "%"}
_BOOLEAN = {"&", "|"}

# Aggregations: name -> result dtype, or None meaning "same as input"
_AGGS: dict[str, DType | None] = {
    "mean": dt.FLOAT64,
    "median": dt.FLOAT64,
    "std": dt.FLOAT64,
    "var": dt.FLOAT64,
    "sum": None,
    "min": None,
    "max": None,
    "first": None,
    "last": None,
    "n_unique": dt.INT64,  # S13
}
_NUMERIC_ONLY_AGGS = {"mean", "median", "std", "var", "sum"}

_STR_FUNCS: dict[str, DType] = {
    "str_detect": dt.BOOL,
    "str_replace": dt.STR,
    "str_to_lower": dt.STR,
    "str_to_upper": dt.STR,
    "str_len": dt.INT64,
}
_NUM_FUNCS: dict[str, DType | None] = {"abs": None, "round": None, "floor": dt.INT64,
                                       "ceiling": dt.INT64, "log": dt.FLOAT64,
                                       "exp": dt.FLOAT64, "sqrt": dt.FLOAT64}
_DT_FUNCS: dict[str, DType] = {"year": dt.INT64, "month": dt.INT64, "day": dt.INT64}

IntoExpr = Union["Expr", int, float, str, bool, _dt.date, _dt.datetime, None]


def lit(value: Any) -> Lit:
    return Lit(value)


def _wrap(v: IntoExpr) -> Expr:
    return v if isinstance(v, Expr) else Lit(v)


@dataclass(frozen=True)
class Expr:
    """Base expression node. Subclasses are the IR; operators build trees."""

    # -- arithmetic ----------------------------------------------------
    def __add__(self, o: IntoExpr) -> BinOp: return BinOp("+", self, _wrap(o))
    def __radd__(self, o: IntoExpr) -> BinOp: return BinOp("+", _wrap(o), self)
    def __sub__(self, o: IntoExpr) -> BinOp: return BinOp("-", self, _wrap(o))
    def __rsub__(self, o: IntoExpr) -> BinOp: return BinOp("-", _wrap(o), self)
    def __mul__(self, o: IntoExpr) -> BinOp: return BinOp("*", self, _wrap(o))
    def __rmul__(self, o: IntoExpr) -> BinOp: return BinOp("*", _wrap(o), self)
    def __truediv__(self, o: IntoExpr) -> BinOp: return BinOp("/", self, _wrap(o))
    def __rtruediv__(self, o: IntoExpr) -> BinOp: return BinOp("/", _wrap(o), self)
    def __floordiv__(self, o: IntoExpr) -> BinOp: return BinOp("//", self, _wrap(o))
    def __mod__(self, o: IntoExpr) -> BinOp: return BinOp("%", self, _wrap(o))
    def __pow__(self, o: IntoExpr) -> Func: return Func("pow", (self, _wrap(o)))
    def __neg__(self) -> UnaryOp: return UnaryOp("-", self)

    # -- comparison / boolean -----------------------------------------
    def __eq__(self, o: object) -> BinOp:  # type: ignore[override]
        return BinOp("==", self, _wrap(o))  # type: ignore[arg-type]
    def __ne__(self, o: object) -> BinOp:  # type: ignore[override]
        return BinOp("!=", self, _wrap(o))  # type: ignore[arg-type]
    def __lt__(self, o: IntoExpr) -> BinOp: return BinOp("<", self, _wrap(o))
    def __le__(self, o: IntoExpr) -> BinOp: return BinOp("<=", self, _wrap(o))
    def __gt__(self, o: IntoExpr) -> BinOp: return BinOp(">", self, _wrap(o))
    def __ge__(self, o: IntoExpr) -> BinOp: return BinOp(">=", self, _wrap(o))
    def __and__(self, o: IntoExpr) -> BinOp: return BinOp("&", self, _wrap(o))
    def __or__(self, o: IntoExpr) -> BinOp: return BinOp("|", self, _wrap(o))
    def __invert__(self) -> UnaryOp: return UnaryOp("!", self)

    def __hash__(self) -> int:  # __eq__ is overridden, so define explicitly
        return hash(repr(self))

    def __bool__(self) -> bool:
        raise ExprTypeError(
            "a dpyr expression is not a Python boolean. Use & | ~ instead of "
            "and/or/not, and is_in() instead of `in`."
        )

    # -- methods -------------------------------------------------------
    def is_na(self) -> Func: return Func("is_na", (self,))
    def is_in(self, values: list[Any] | tuple[Any, ...]) -> Func:
        return Func("is_in", (self, Lit(tuple(values))))
    def cast(self, to: DType) -> Cast: return Cast(self, to)
    def between(self, lo: IntoExpr, hi: IntoExpr) -> BinOp:
        return (self >= lo) & (self <= hi)

    def _agg(self, name: str, na_rm: bool = True) -> Agg:
        return Agg(name, self, na_rm)

    def mean(self, na_rm: bool = True) -> Agg: return self._agg("mean", na_rm)
    def median(self, na_rm: bool = True) -> Agg: return self._agg("median", na_rm)
    def sum(self, na_rm: bool = True) -> Agg: return self._agg("sum", na_rm)
    def min(self, na_rm: bool = True) -> Agg: return self._agg("min", na_rm)
    def max(self, na_rm: bool = True) -> Agg: return self._agg("max", na_rm)
    def std(self, na_rm: bool = True) -> Agg: return self._agg("std", na_rm)
    def var(self, na_rm: bool = True) -> Agg: return self._agg("var", na_rm)
    def first(self) -> Agg: return self._agg("first")
    def last(self) -> Agg: return self._agg("last")
    def n_unique(self) -> Agg: return self._agg("n_unique")

    def abs(self) -> Func: return Func("abs", (self,))
    def round(self, digits: int = 0) -> Func: return Func("round", (self, Lit(digits)))
    def floor(self) -> Func: return Func("floor", (self,))
    def ceiling(self) -> Func: return Func("ceiling", (self,))
    def log(self) -> Func: return Func("log", (self,))
    def exp(self) -> Func: return Func("exp", (self,))
    def sqrt(self) -> Func: return Func("sqrt", (self,))

    def str_detect(self, pattern: str) -> Func:
        return Func("str_detect", (self, Lit(pattern)))
    def str_replace(self, pattern: str, replacement: str) -> Func:
        return Func("str_replace", (self, Lit(pattern), Lit(replacement)))
    def str_to_lower(self) -> Func: return Func("str_to_lower", (self,))
    def str_to_upper(self) -> Func: return Func("str_to_upper", (self,))
    def str_len(self) -> Func: return Func("str_len", (self,))

    def year(self) -> Func: return Func("year", (self,))
    def month(self) -> Func: return Func("month", (self,))
    def day(self) -> Func: return Func("day", (self,))


@dataclass(frozen=True, eq=False)
class Col(Expr):
    name: str

    def __repr__(self) -> str:
        return f"col.{self.name}"


@dataclass(frozen=True, eq=False)
class Lit(Expr):
    value: Any

    def __repr__(self) -> str:
        return f"lit({self.value!r})"


@dataclass(frozen=True, eq=False)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"


@dataclass(frozen=True, eq=False)
class UnaryOp(Expr):
    op: str
    operand: Expr

    def __repr__(self) -> str:
        return f"({self.op}{self.operand!r})"


@dataclass(frozen=True, eq=False)
class Func(Expr):
    name: str
    args: tuple[Expr, ...]

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(map(repr, self.args))})"


@dataclass(frozen=True, eq=False)
class Cast(Expr):
    operand: Expr
    to: DType

    def __repr__(self) -> str:
        return f"cast({self.operand!r}, {self.to!r})"


@dataclass(frozen=True, eq=False)
class Agg(Expr):
    name: str
    operand: Expr
    na_rm: bool = True

    def __repr__(self) -> str:
        return f"{self.operand!r}.{self.name}(na_rm={self.na_rm})"


@dataclass(frozen=True, eq=False)
class N(Expr):
    """n() — group size. S13: Int64."""

    def __repr__(self) -> str:
        return "n()"


@dataclass(frozen=True, eq=False)
class Desc(Expr):
    """Sort-direction marker; only valid directly inside arrange()."""
    operand: Expr

    def __repr__(self) -> str:
        return f"desc({self.operand!r})"


@dataclass(frozen=True, eq=False)
class IfElse(Expr):
    cond: Expr
    true: Expr
    false: Expr

    def __repr__(self) -> str:
        return f"if_else({self.cond!r}, {self.true!r}, {self.false!r})"


@dataclass(frozen=True, eq=False)
class CaseWhen(Expr):
    cases: tuple[tuple[Expr, Expr], ...]
    default: Expr = field(default_factory=lambda: Lit(None))

    def __repr__(self) -> str:
        body = ", ".join(f"{c!r}: {v!r}" for c, v in self.cases)
        return f"case_when({body}, default={self.default!r})"


# ---------------------------------------------------------------------
# public constructors


class _ColProxy:
    """The global `col` proxy: `col.height` -> Col('height')."""

    def __getattr__(self, name: str) -> Col:
        if name.startswith("_"):
            raise AttributeError(name)
        return Col(name)

    def __getitem__(self, name: str) -> Col:
        return Col(name)


col = _ColProxy()


def n() -> N:
    return N()


def desc(e: Expr) -> Desc:
    if not isinstance(e, Expr):
        raise ExprTypeError(f"desc() expects an expression, got {type(e).__name__}")
    return Desc(e)


def if_else(cond: Expr, true: IntoExpr, false: IntoExpr) -> IfElse:
    return IfElse(cond, _wrap(true), _wrap(false))


def case_when(*cases: tuple[Expr, IntoExpr], default: IntoExpr = None) -> CaseWhen:
    if not cases:
        raise ExprTypeError("case_when() needs at least one (condition, value) pair")
    return CaseWhen(tuple((c, _wrap(v)) for c, v in cases), _wrap(default))


# ---------------------------------------------------------------------
# type inference (the schema-eager engine)


def _lit_dtype(value: Any) -> DType:
    if value is None:
        return dt.NULL
    if isinstance(value, bool):
        return dt.BOOL
    if isinstance(value, int):
        return dt.INT64
    if isinstance(value, float):
        return dt.FLOAT64
    if isinstance(value, str):
        return dt.STR
    if isinstance(value, _dt.datetime):
        return dt.DATETIME
    if isinstance(value, _dt.date):
        return dt.DATE
    if isinstance(value, tuple):  # is_in value set
        return dt.NULL
    raise ExprTypeError(f"unsupported literal type: {type(value).__name__}")


def infer_dtype(expr: Expr, schema: Schema, *, in_agg: bool = False,
                context: str = "expression") -> DType:
    """Validate `expr` against `schema` and return its dtype.

    Raises ColumnNotFoundError / ExprTypeError immediately — this is what
    makes verbs feel eager even though no data moves.
    """
    def rec(e: Expr, in_agg: bool) -> DType:
        if isinstance(e, Col):
            if e.name not in schema:
                raise ColumnNotFoundError(e.name, schema, context)
            return schema[e.name]
        if isinstance(e, Lit):
            return _lit_dtype(e.value)
        if isinstance(e, N):
            return dt.INT64
        if isinstance(e, Desc):
            raise ExprTypeError("desc() is only valid directly inside arrange()")
        if isinstance(e, Cast):
            rec(e.operand, in_agg)
            return e.to
        if isinstance(e, Agg):
            if in_agg:
                raise ExprTypeError(f"nested aggregation in {e!r}")
            inner = rec(e.operand, True)
            if e.name == "sum" and inner == dt.BOOL:
                return dt.INT64  # sum(bool) counts trues, like R
            if e.name in _NUMERIC_ONLY_AGGS and not (dt.is_numeric(inner) or inner == dt.NULL):
                raise ExprTypeError(
                    f".{e.name}() needs a numeric column, got {inner!r} in {e!r}")
            result = _AGGS[e.name]
            return result if result is not None else inner
        if isinstance(e, UnaryOp):
            inner = rec(e.operand, in_agg)
            if e.op == "-":
                if not (dt.is_numeric(inner) or inner == dt.NULL):
                    raise ExprTypeError(f"unary - needs a numeric, got {inner!r}")
                return inner
            if e.op == "!":
                if inner not in (dt.BOOL, dt.NULL):
                    raise ExprTypeError(f"~ needs a boolean, got {inner!r}")
                return dt.BOOL
            raise AssertionError(e.op)
        if isinstance(e, BinOp):
            lt, rt = rec(e.left, in_agg), rec(e.right, in_agg)
            if e.op in _ARITHMETIC:
                out = dt.arith_result(e.op, lt, rt)
                if out is None:
                    raise ExprTypeError(f"cannot apply {e.op} to {lt!r} and {rt!r} in {e!r}")
                return out
            if e.op in _COMPARISONS:
                if dt.unify(lt, rt) is None:
                    raise ExprTypeError(f"cannot compare {lt!r} with {rt!r} in {e!r}")
                return dt.BOOL
            if e.op in _BOOLEAN:
                for side in (lt, rt):
                    if side not in (dt.BOOL, dt.NULL):
                        raise ExprTypeError(f"{e.op} needs booleans, got {side!r} in {e!r}")
                return dt.BOOL
            raise AssertionError(e.op)
        if isinstance(e, IfElse):
            ct = rec(e.cond, in_agg)
            if ct not in (dt.BOOL, dt.NULL):
                raise ExprTypeError(f"if_else() condition must be boolean, got {ct!r}")
            tt, ft = rec(e.true, in_agg), rec(e.false, in_agg)
            out = dt.unify(tt, ft)
            if out is None:
                raise ExprTypeError(
                    f"if_else() branches have incompatible dtypes {tt!r} and {ft!r}")
            return out
        if isinstance(e, CaseWhen):
            acc: DType = dt.NULL
            for cond, value in e.cases:
                ct = rec(cond, in_agg)
                if ct not in (dt.BOOL, dt.NULL):
                    raise ExprTypeError(
                        f"case_when() condition must be boolean, got {ct!r} in {cond!r}")
                vt = rec(value, in_agg)
                unified = dt.unify(acc, vt)
                if unified is None:
                    raise ExprTypeError(
                        f"case_when() values have incompatible dtypes {acc!r} and {vt!r}")
                acc = unified
            dt_default = rec(e.default, in_agg)
            final = dt.unify(acc, dt_default)
            if final is None:
                raise ExprTypeError(
                    f"case_when() default dtype {dt_default!r} incompatible with {acc!r}")
            return final
        if isinstance(e, Func):
            arg_types = [rec(a, in_agg) for a in e.args]
            return _func_dtype(e, arg_types)
        raise AssertionError(f"unknown expression node {type(e).__name__}")

    return rec(expr, in_agg)


def _func_dtype(e: Func, arg_types: list[DType]) -> DType:
    name, first = e.name, arg_types[0]
    if name == "is_na":
        return dt.BOOL
    if name == "is_in":
        return dt.BOOL
    if name == "pow":
        out = dt.arith_result("*", arg_types[0], arg_types[1])
        if out is None:
            raise ExprTypeError(f"** needs numerics, got {arg_types[0]!r}, {arg_types[1]!r}")
        return dt.FLOAT64
    if name in _STR_FUNCS:
        if first not in (dt.STR, dt.NULL):
            raise ExprTypeError(f".{name}() needs a string column, got {first!r} in {e!r}")
        return _STR_FUNCS[name]
    if name in _NUM_FUNCS:
        if not (dt.is_numeric(first) or first == dt.NULL):
            raise ExprTypeError(f".{name}() needs a numeric column, got {first!r} in {e!r}")
        out = _NUM_FUNCS[name]
        return out if out is not None else first
    if name in _DT_FUNCS:
        if first not in (*dt.TEMPORAL, dt.NULL):
            raise ExprTypeError(f".{name}() needs a date/datetime, got {first!r} in {e!r}")
        return _DT_FUNCS[name]
    raise AssertionError(f"unknown function {name}")


def contains_agg(e: Expr) -> bool:
    """True if the expression aggregates (Agg or n()) at any depth."""
    if isinstance(e, (Agg, N)):
        return True
    children: tuple[Expr, ...]
    if isinstance(e, BinOp):
        children = (e.left, e.right)
    elif isinstance(e, (UnaryOp, Desc)):
        children = (e.operand,)
    elif isinstance(e, Cast):
        children = (e.operand,)
    elif isinstance(e, Func):
        children = e.args
    elif isinstance(e, IfElse):
        children = (e.cond, e.true, e.false)
    elif isinstance(e, CaseWhen):
        children = tuple(x for pair in e.cases for x in pair) + (e.default,)
    else:
        children = ()
    return any(contains_agg(c) for c in children)
