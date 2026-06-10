"""IR -> polars LazyFrame compiler (Epic 2).

Semantics shims applied here so polars output matches the conformance spec:
- S3  sorts are stable with nulls last (maintain_order + nulls_last)
- S4  `/` casts both sides to Float64 (int/int -> float, /0 -> inf, S14)
- S7  grouped summarize output is sorted by group keys
- S9  handled in the plan layer (groups metadata)
- S13 counts (n, n_unique, str_len, ...) are cast to Int64
- S1  is_na on floats is null-or-NaN (R's is.na(NaN) is TRUE)
"""

from __future__ import annotations

from typing import Any

import polars as pl

from . import dtypes as dt
from . import plan as p
from .backend import PolarsPayload, resolve
from .dtypes import DType
from .errors import DpyrError
from .expr import (
    Agg,
    BinOp,
    Cast,
    CaseWhen,
    Col,
    Desc,
    Expr,
    Func,
    IfElse,
    Lit,
    N,
    UnaryOp,
    Window,
    contains_agg,
    contains_window,
    infer_dtype,
)

PL_DTYPE: dict[DType, Any] = {
    dt.INT64: pl.Int64,
    dt.FLOAT64: pl.Float64,
    dt.BOOL: pl.Boolean,
    dt.STR: pl.String,
    dt.DATE: pl.Date,
    dt.DATETIME: pl.Datetime("us"),
}
FROM_PL: dict[Any, DType] = {
    pl.Int8: dt.INT64, pl.Int16: dt.INT64, pl.Int32: dt.INT64, pl.Int64: dt.INT64,
    pl.UInt8: dt.INT64, pl.UInt16: dt.INT64, pl.UInt32: dt.INT64, pl.UInt64: dt.INT64,
    pl.Float32: dt.FLOAT64, pl.Float64: dt.FLOAT64,
    pl.Boolean: dt.BOOL, pl.String: dt.STR, pl.Utf8: dt.STR,
    pl.Date: dt.DATE, pl.Null: dt.NULL,
}


def schema_from_polars(lf: pl.LazyFrame) -> dict[str, DType]:
    out: dict[str, DType] = {}
    for name, pld in lf.collect_schema().items():
        if isinstance(pld, pl.Datetime):
            out[name] = dt.DATETIME
        elif pld in FROM_PL:
            out[name] = FROM_PL[pld]
        else:
            raise DpyrError(
                f"column '{name}' has unsupported dtype {pld}; supported: "
                "ints, floats, bool, string, date, datetime")
    return out


def _normalize(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Coerce a user frame to canonical dtypes (Int64/Float64/...)."""
    casts = []
    for name, pld in lf.collect_schema().items():
        if pld in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16,
                   pl.UInt32, pl.UInt64):
            casts.append(pl.col(name).cast(pl.Int64))
        elif pld == pl.Float32:
            casts.append(pl.col(name).cast(pl.Float64))
        elif isinstance(pld, pl.Datetime) and pld != pl.Datetime("us"):
            casts.append(pl.col(name).cast(pl.Datetime("us")))
    return lf.with_columns(casts) if casts else lf


class _Ctx:
    """Compilation context: schema for dtype queries, window for grouped
    row-level ops (mutate/filter on grouped frames)."""

    def __init__(self, schema: dict[str, DType], window: tuple[str, ...] = ()):
        self.schema = schema
        self.window = window


def compile_expr(e: Expr, ctx: _Ctx) -> pl.Expr:
    out = _compile(e, ctx)
    if ctx.window and (contains_agg(e) or contains_window(e)):
        out = out.over(list(ctx.window))
    return out


def _compile(e: Expr, ctx: _Ctx) -> pl.Expr:
    if isinstance(e, Col):
        return pl.col(e.name)
    if isinstance(e, Lit):
        if isinstance(e.value, bool):
            return pl.lit(e.value, dtype=pl.Boolean)
        if isinstance(e.value, int):
            return pl.lit(e.value, dtype=pl.Int64)  # not Int32
        if isinstance(e.value, float):
            return pl.lit(e.value, dtype=pl.Float64)
        return pl.lit(e.value)
    if isinstance(e, N):
        return pl.len().cast(pl.Int64)
    if isinstance(e, Cast):
        return _compile(e.operand, ctx).cast(PL_DTYPE[e.to])
    if isinstance(e, Desc):
        raise AssertionError("Desc reaches the compiler only inside Arrange")
    if isinstance(e, UnaryOp):
        inner = _compile(e.operand, ctx)
        return -inner if e.op == "-" else ~inner
    if isinstance(e, BinOp):
        lhs, rhs = _compile(e.left, ctx), _compile(e.right, ctx)
        if e.op == "/":  # S4/S14: float division, div-by-zero -> inf
            return lhs.cast(pl.Float64) / rhs.cast(pl.Float64)
        ops = {
            "+": lhs.__add__, "-": lhs.__sub__, "*": lhs.__mul__,
            "//": lhs.__floordiv__, "%": lhs.__mod__,
            "==": lhs.__eq__, "!=": lhs.__ne__, "<": lhs.__lt__,
            "<=": lhs.__le__, ">": lhs.__gt__, ">=": lhs.__ge__,
            "&": lhs.__and__, "|": lhs.__or__,
        }
        return ops[e.op](rhs)
    if isinstance(e, Agg):
        inner = _compile(e.operand, ctx)
        compiled = _compile_agg(e, inner, ctx)
        if not e.na_rm and e.name not in ("first", "last", "n_unique"):
            compiled = (
                pl.when(_compile(e.operand, ctx).is_null().any())
                .then(pl.lit(None))
                .otherwise(compiled)
            )
        return compiled
    if isinstance(e, IfElse):
        return (pl.when(_compile(e.cond, ctx))
                .then(_compile(e.true, ctx))
                .otherwise(_compile(e.false, ctx)))
    if isinstance(e, CaseWhen):
        cases = iter(e.cases)
        cond, val = next(cases)
        out: Any = pl.when(_compile(cond, ctx)).then(_compile(val, ctx))
        for cond, val in cases:
            out = out.when(_compile(cond, ctx)).then(_compile(val, ctx))
        return out.otherwise(_compile(e.default, ctx))
    if isinstance(e, Window):
        return _compile_window(e, ctx)
    if isinstance(e, Func):
        return _compile_func(e, ctx)
    raise AssertionError(f"unhandled expression node {type(e).__name__}")


def _compile_window(e: Window, ctx: _Ctx) -> pl.Expr:
    if e.name == "row_number":
        return (pl.int_range(pl.len()) + 1).cast(pl.Int64)
    assert e.operand is not None
    x = _compile(e.operand, ctx)
    if e.name in ("lag", "lead"):
        n = e.n if e.name == "lag" else -e.n
        if isinstance(e.default, Lit) and e.default.value is None:
            return x.shift(n)
        return x.shift(n, fill_value=_compile(e.default, ctx))
    if e.name == "min_rank":
        return x.rank(method="min", descending=e.descending).cast(pl.Int64)
    if e.name == "dense_rank":
        return x.rank(method="dense", descending=e.descending).cast(pl.Int64)
    if e.name == "percent_rank":
        # dplyr: (min_rank - 1) / (number of non-missing - 1)
        rank = x.rank(method="min", descending=e.descending).cast(pl.Float64)
        denom = (x.is_not_null().sum() - 1).cast(pl.Float64)
        return (rank - 1) / denom
    if e.name == "cum_sum":
        in_dt = infer_dtype(e.operand, ctx.schema)
        base = x.cast(pl.Int64) if in_dt == dt.BOOL else x
        return base.cum_sum()
    if e.name == "cum_min":
        return x.cum_min()
    if e.name == "cum_max":
        return x.cum_max()
    raise AssertionError(e.name)


def _compile_agg(e: Agg, inner: pl.Expr, ctx: _Ctx) -> pl.Expr:
    name = e.name
    if name == "mean":
        return inner.mean()
    if name == "median":
        return inner.median()
    if name == "sum":
        in_dt = infer_dtype(e.operand, ctx.schema)
        out = inner.sum()
        return out.cast(pl.Int64) if in_dt == dt.BOOL else out
    if name == "min":
        return inner.min()
    if name == "max":
        return inner.max()
    if name == "std":
        return inner.std(ddof=1)
    if name == "var":
        return inner.var(ddof=1)
    if name == "first":
        return inner.first()
    if name == "last":
        return inner.last()
    if name == "n_unique":
        return inner.n_unique().cast(pl.Int64)
    raise AssertionError(name)


def _compile_func(e: Func, ctx: _Ctx) -> pl.Expr:
    args = [_compile(a, ctx) for a in e.args]
    x = args[0]
    name = e.name
    if name == "coalesce":
        return pl.coalesce(args)
    if name == "replace_na":
        return x.fill_null(args[1])
    if name == "is_na":
        # S1: R's is.na is TRUE for NaN too
        if infer_dtype(e.args[0], ctx.schema) == dt.FLOAT64:
            return x.is_null() | x.is_nan()
        return x.is_null()
    if name == "is_in":
        values = e.args[1]
        assert isinstance(values, Lit)
        return x.is_in(list(values.value))
    if name == "pow":
        return x.cast(pl.Float64).pow(args[1].cast(pl.Float64))
    if name == "abs":
        return x.abs()
    if name == "round":
        digits = e.args[1]
        assert isinstance(digits, Lit)
        return x.round(int(digits.value))
    if name == "floor":
        return x.floor().cast(pl.Int64)
    if name == "ceiling":
        return x.ceil().cast(pl.Int64)
    if name == "log":
        return x.cast(pl.Float64).log()
    if name == "exp":
        return x.cast(pl.Float64).exp()
    if name == "sqrt":
        return x.cast(pl.Float64).sqrt()
    if name == "str_detect":
        return x.str.contains(_lit_str(e.args[1]))
    if name == "str_replace":
        return x.str.replace(_lit_str(e.args[1]), _lit_str(e.args[2]))
    if name == "str_to_lower":
        return x.str.to_lowercase()
    if name == "str_to_upper":
        return x.str.to_uppercase()
    if name == "str_len":
        return x.str.len_chars().cast(pl.Int64)
    if name == "year":
        return x.dt.year().cast(pl.Int64)
    if name == "month":
        return x.dt.month().cast(pl.Int64)
    if name == "day":
        return x.dt.day().cast(pl.Int64)
    raise AssertionError(f"unhandled function {name}")


def _lit_str(e: Expr) -> str:
    assert isinstance(e, Lit)
    return str(e.value)


def compile_plan(node: p.PlanNode) -> pl.LazyFrame:
    if isinstance(node, p.Source):
        payload = resolve(node.token)
        assert isinstance(payload, PolarsPayload)
        return payload.lf

    if isinstance(node, p.Filter):
        lf = compile_plan(node.child)
        ctx = _Ctx(node.child.schema, node.child.groups)
        pred = compile_expr(node.predicates[0], ctx)
        for q in node.predicates[1:]:
            pred = pred & compile_expr(q, ctx)
        return lf.filter(pred)

    if isinstance(node, p.Mutate):
        lf = compile_plan(node.child)
        schema = dict(node.child.schema)
        for name, e in node.exprs:
            ctx = _Ctx(schema, node.child.groups)
            lf = lf.with_columns(compile_expr(e, ctx).alias(name))
            schema[name] = infer_dtype(e, schema)
        return lf

    if isinstance(node, p.Select):
        return compile_plan(node.child).select(list(node.schema))

    if isinstance(node, p.Rename):
        return compile_plan(node.child).rename(
            {old: new for new, old in node.mapping})

    if isinstance(node, p.Arrange):
        lf = compile_plan(node.child)
        ctx = _Ctx(node.child.schema, ())
        by, descending = [], []
        for k in node.keys:
            inner = k.operand if isinstance(k, Desc) else k
            by.append(compile_expr(inner, ctx))
            descending.append(isinstance(k, Desc))
        return lf.sort(by, descending=descending, nulls_last=True,
                       maintain_order=True)  # S3

    if isinstance(node, p.Distinct):
        lf = compile_plan(node.child)
        if node.cols:
            lf = lf.select(list(node.schema))
        return lf.unique(keep="first", maintain_order=True)

    if isinstance(node, p.Slice):
        groups = node.child.groups
        lf = compile_plan(node.child)
        if node.kind == "sample":
            if groups:
                raise DpyrError("slice_sample() on grouped dataframes is not supported yet")
            # LCG-mix sampling, identical on both engines (S33); the seed
            # is mixed before multiplying so it changes the permutation
            seed = node.seed if node.seed is not None else 0
            idx = pl.int_range(pl.len(), dtype=pl.Int64)
            a = (idx + seed) % 2147483647
            b = (a * 48271) % 2147483647
            key = ((b * 48271) % 2147483647).alias("__dpyr_key")
            return (lf.with_columns(key, idx.alias("__dpyr_idx"))
                    .sort(["__dpyr_key", "__dpyr_idx"])
                    .head(node.n).drop(["__dpyr_key", "__dpyr_idx"]))
        if groups:
            gb = lf.group_by(list(groups), maintain_order=True)
            out = gb.head(node.n) if node.kind == "head" else gb.tail(node.n)
            return out.select(list(node.schema))  # keys are moved first; undo
        return lf.head(node.n) if node.kind == "head" else lf.tail(node.n)

    if isinstance(node, p.Separate):
        lf = compile_plan(node.child)
        parts = pl.col(node.column).str.split(node.sep)
        new_cols = [parts.list.get(i, null_on_oob=True).alias(name)
                    for i, name in enumerate(node.into)]
        return lf.with_columns(new_cols).select(list(node.schema))

    if isinstance(node, p.Unite):
        lf = compile_plan(node.child)
        pieces: list[pl.Expr] = []
        for c in node.cols:
            piece = pl.col(c).cast(pl.String)
            if not node.na_rm:
                piece = piece.fill_null("NA")  # tidyr renders missing as 'NA'
            pieces.append(piece)
        united = pl.concat_str(pieces, separator=node.sep,
                               ignore_nulls=node.na_rm).alias(node.new)
        if node.na_rm:
            united = united.fill_null("")  # all-missing row joins to ''
        return lf.with_columns(united).select(list(node.schema))

    if isinstance(node, (p.GroupBy, p.Ungroup)):
        return compile_plan(node.child)  # grouping is metadata

    if isinstance(node, p.Summarize):
        lf = compile_plan(node.child)
        ctx = _Ctx(node.child.schema, ())
        aggs = [compile_expr(e, ctx).alias(name) for name, e in node.aggs]
        keys = list(node.child.groups)
        if not keys:
            return lf.select(aggs)
        out = lf.group_by(keys).agg(aggs)
        return out.sort(keys, nulls_last=True, maintain_order=False)  # S7

    if isinstance(node, p.Join):
        left = compile_plan(node.left)
        right = compile_plan(node.right)
        if node.how in ("semi", "anti"):
            return left.join(right, on=list(node.on), how=node.how,
                             nulls_equal=node.na_matches == "na")
        overlap = ((set(node.left.schema) & set(node.right.schema))
                   - set(node.on))
        sx, sy = node.suffix
        if overlap:
            left = left.rename({c: c + sx for c in overlap})
            right = right.rename({c: c + sy for c in overlap})
        out = left.join(right, on=list(node.on), how=node.how,
                        coalesce=True, nulls_equal=node.na_matches == "na")
        return out.select(list(node.schema))

    if isinstance(node, p.PivotLonger):
        lf = compile_plan(node.child)
        index = [c for c in node.child.schema if c not in node.cols]
        out = lf.unpivot(on=list(node.cols), index=index,
                         variable_name=node.names_to,
                         value_name=node.values_to)
        target = node.schema[node.values_to]
        if target != dt.NULL:
            out = out.with_columns(pl.col(node.values_to).cast(PL_DTYPE[target]))
        return out.select(list(node.schema))

    if isinstance(node, p.PivotWider):
        df = compile_plan(node.child).collect()
        index = [c for c in node.child.schema
                 if c not in (node.names_from, node.values_from)]
        wide = df.pivot(on=node.names_from, values=node.values_from,
                        index=index, aggregate_function="first")
        return wide.lazy()

    raise AssertionError(f"unhandled plan node {type(node).__name__}")
