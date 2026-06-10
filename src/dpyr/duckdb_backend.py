"""IR -> SQL compiler for duckdb (Epic 5).

Strategy: each plan node wraps its child in a subquery; duckdb's optimizer
flattens the nesting. Ordering is tracked out-of-band (SQL subqueries don't
guarantee order) and applied once at the outermost query — the dbplyr
approach.

Semantics shims to match polars/the conformance spec:
- S3   ORDER BY ... NULLS LAST, stable via row-number tiebreak
- S4   `/` casts to DOUBLE (int/int -> float; x/0 -> inf with ieee ops)
- S7   grouped summarize ordered by keys NULLS LAST
- S10  na_matches="na" joins use IS NOT DISTINCT FROM
- S13  counts cast to BIGINT
- n_unique counts NULL as a distinct value (like polars/dplyr n_distinct)
- na_rm=False emulated with CASE WHEN count(*) <> count(x)
- %    floor-mod (R/Python), not SQL trunc-mod
"""

from __future__ import annotations

import datetime as _dt
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import dtypes as dt
from . import plan as p
from .backend import DuckPayload, resolve, sources_of
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
    contains_agg,
)

if TYPE_CHECKING:
    import duckdb
    import polars as pl

SQL_DTYPE: dict[DType, str] = {
    dt.INT64: "BIGINT",
    dt.FLOAT64: "DOUBLE",
    dt.BOOL: "BOOLEAN",
    dt.STR: "VARCHAR",
    dt.DATE: "DATE",
    dt.DATETIME: "TIMESTAMP",
}

_DUCK_TO_DTYPE = {
    "TINYINT": dt.INT64, "SMALLINT": dt.INT64, "INTEGER": dt.INT64,
    "BIGINT": dt.INT64, "HUGEINT": dt.INT64, "UTINYINT": dt.INT64,
    "USMALLINT": dt.INT64, "UINTEGER": dt.INT64, "UBIGINT": dt.INT64,
    "FLOAT": dt.FLOAT64, "DOUBLE": dt.FLOAT64,
    "BOOLEAN": dt.BOOL, "VARCHAR": dt.STR,
    "DATE": dt.DATE, "TIMESTAMP": dt.DATETIME,
    '"NULL"': dt.NULL, "NULL": dt.NULL,
}


def schema_from_duckdb(con: duckdb.DuckDBPyConnection, table_sql: str) -> dict[str, DType]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {table_sql}").fetchall()
    out: dict[str, DType] = {}
    for name, typ, *_ in rows:
        base = typ.split("(")[0].upper()
        if base.startswith("DECIMAL"):
            out[name] = dt.FLOAT64
        elif base in _DUCK_TO_DTYPE:
            out[name] = _DUCK_TO_DTYPE[base]
        else:
            raise DpyrError(f"column '{name}' has unsupported duckdb type {typ}")
    return out


def q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def sql_lit(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, _dt.datetime):
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if isinstance(value, _dt.date):
        return f"DATE '{value.isoformat()}'"
    raise DpyrError(f"unsupported literal for SQL: {value!r}")


_cte_counter = itertools.count()


def _cte_name() -> str:
    return f"__dpyr_w{next(_cte_counter)}"


@dataclass
class _Ctx:
    schema: dict[str, DType]
    window: tuple[str, ...] = ()  # PARTITION BY for grouped mutate/filter
    in_agg: bool = False          # inside summarize aggregation list


def compile_expr(e: Expr, ctx: _Ctx) -> str:
    if isinstance(e, Col):
        return q(e.name)
    if isinstance(e, Lit):
        return sql_lit(e.value)
    if isinstance(e, N):
        over = _over(ctx)
        return f"CAST(count(*){over} AS BIGINT)"
    if isinstance(e, Cast):
        return f"CAST({compile_expr(e.operand, ctx)} AS {SQL_DTYPE[e.to]})"
    if isinstance(e, Desc):
        raise AssertionError("Desc reaches the compiler only inside Arrange")
    if isinstance(e, UnaryOp):
        inner = compile_expr(e.operand, ctx)
        return f"(-{inner})" if e.op == "-" else f"(NOT {inner})"
    if isinstance(e, BinOp):
        lhs, rhs = compile_expr(e.left, ctx), compile_expr(e.right, ctx)
        if e.op == "/":  # S4/S14
            return f"(CAST({lhs} AS DOUBLE) / CAST({rhs} AS DOUBLE))"
        if e.op == "//":
            from .expr import infer_dtype
            lt = infer_dtype(e.left, ctx.schema)
            rt = infer_dtype(e.right, ctx.schema)
            body = f"FLOOR(CAST({lhs} AS DOUBLE) / CAST({rhs} AS DOUBLE))"
            if lt == dt.INT64 and rt == dt.INT64:
                return f"CAST({body} AS BIGINT)"
            return body
        if e.op == "%":  # floor-mod like R/Python
            return f"((({lhs}) % ({rhs}) + ({rhs})) % ({rhs}))"
        op = {"==": "=", "!=": "<>", "&": "AND", "|": "OR"}.get(e.op, e.op)
        return f"({lhs} {op} {rhs})"
    if isinstance(e, Agg):
        return _compile_agg(e, ctx)
    if isinstance(e, IfElse):
        return (f"CASE WHEN {compile_expr(e.cond, ctx)} "
                f"THEN {compile_expr(e.true, ctx)} "
                f"ELSE {compile_expr(e.false, ctx)} END")
    if isinstance(e, CaseWhen):
        whens = " ".join(
            f"WHEN {compile_expr(c, ctx)} THEN {compile_expr(v, ctx)}"
            for c, v in e.cases)
        return f"CASE {whens} ELSE {compile_expr(e.default, ctx)} END"
    if isinstance(e, Func):
        return _compile_func(e, ctx)
    raise AssertionError(f"unhandled expression node {type(e).__name__}")


def _over(ctx: _Ctx) -> str:
    if ctx.in_agg or not ctx.window:
        return ""
    keys = ", ".join(q(k) for k in ctx.window)
    return f" OVER (PARTITION BY {keys})"


def _compile_agg(e: Agg, ctx: _Ctx) -> str:
    x = compile_expr(e.operand, ctx)
    over = _over(ctx)
    from .expr import infer_dtype
    in_dt = infer_dtype(e.operand, ctx.schema)
    fn = {
        "mean": "avg", "median": "median", "sum": "sum", "min": "min",
        "max": "max", "std": "stddev_samp", "var": "var_samp",
        "first": "first", "last": "last",
    }.get(e.name)
    if e.name == "n_unique":
        # polars/dplyr count NULL as a distinct value; SQL doesn't
        body = (f"(count(DISTINCT {x}){over} + "
                f"CASE WHEN count(*){over} <> count({x}){over} THEN 1 ELSE 0 END)")
        return f"CAST({body} AS BIGINT)"
    assert fn is not None, e.name
    if e.name == "sum" and in_dt == dt.BOOL:
        body = f"sum(CASE WHEN {x} THEN 1 WHEN {x} IS NULL THEN NULL ELSE 0 END){over}"
        body = f"COALESCE({body}, 0)" if e.na_rm else body
        result = f"CAST({body} AS BIGINT)"
        if not e.na_rm:
            return (f"CASE WHEN count(*){over} <> count({x}){over} "
                    f"THEN NULL ELSE {result} END")
        return result
    body = f"{fn}({x}){over}"
    if e.name == "sum" and e.na_rm:
        zero = "0.0" if in_dt == dt.FLOAT64 else "0"
        body = f"COALESCE({body}, {zero})"  # R: sum of nothing is 0
    if e.name in ("mean", "median", "std", "var"):
        body = f"CAST({body} AS DOUBLE)"
    if not e.na_rm and e.name not in ("first", "last"):
        return (f"CASE WHEN count(*){over} <> count({x}){over} "
                f"THEN NULL ELSE {body} END")
    return body


def _compile_func(e: Func, ctx: _Ctx) -> str:
    if e.name == "is_in":  # second arg is a value tuple, not an expression
        x = compile_expr(e.args[0], ctx)
        values = e.args[1]
        assert isinstance(values, Lit)
        items = ", ".join(sql_lit(v) for v in values.value)
        return f"({x} IN ({items}))"
    args = [compile_expr(a, ctx) for a in e.args]
    x = args[0]
    name = e.name
    from .expr import infer_dtype
    if name == "is_na":
        if infer_dtype(e.args[0], ctx.schema) == dt.FLOAT64:
            return f"({x} IS NULL OR isnan({x}))"
        return f"({x} IS NULL)"
    if name == "pow":
        return f"pow(CAST({x} AS DOUBLE), CAST({args[1]} AS DOUBLE))"
    if name == "abs":
        return f"abs({x})"
    if name == "round":
        return f"round({x}, {args[1]})"
    if name == "floor":
        return f"CAST(floor({x}) AS BIGINT)"
    if name == "ceiling":
        return f"CAST(ceil({x}) AS BIGINT)"
    if name in ("log", "exp", "sqrt"):
        fn = {"log": "ln", "exp": "exp", "sqrt": "sqrt"}[name]
        return f"{fn}(CAST({x} AS DOUBLE))"
    if name == "str_detect":
        return f"regexp_matches({x}, {args[1]})"
    if name == "str_replace":
        return f"regexp_replace({x}, {args[1]}, {args[2]})"
    if name == "str_to_lower":
        return f"lower({x})"
    if name == "str_to_upper":
        return f"upper({x})"
    if name == "str_len":
        return f"CAST(length({x}) AS BIGINT)"
    if name in ("year", "month", "day"):
        return f"CAST({name}({x}) AS BIGINT)"
    raise AssertionError(f"unhandled function {name}")


# ---------------------------------------------------------------------
# plan compilation: returns (subquery_sql, order_by or None)


@dataclass
class _Compiled:
    sql: str          # a SELECT statement (no outer ORDER BY)
    order: str | None  # ORDER BY clause to apply at the outermost level


def _cols(schema: dict[str, DType]) -> str:
    return ", ".join(q(c) for c in schema)


def compile_plan(node: p.PlanNode) -> _Compiled:
    if isinstance(node, p.Source):
        payload = resolve(node.token)
        assert isinstance(payload, DuckPayload)
        return _Compiled(f"SELECT {_cols(node.schema)} FROM {payload.table_sql}", None)

    if isinstance(node, p.Filter):
        c = compile_plan(node.child)
        ctx = _Ctx(node.child.schema, node.child.groups)
        if node.child.groups and any(contains_agg(pr) for pr in node.predicates):
            # window functions can't appear in WHERE: project, then filter.
            # MATERIALIZED CTE works around duckdb's window-over-window bug
            # (unordered_map::at in 1.5.x).
            preds = [compile_expr(pr, ctx) for pr in node.predicates]
            flags = ", ".join(f"{s} AS {q(f'__pred{i}')}" for i, s in enumerate(preds))
            cte = _cte_name()
            inner = (f"WITH {q(cte)} AS MATERIALIZED ({c.sql}) "
                     f"SELECT *, {flags} FROM {q(cte)}")
            conds = " AND ".join(q(f"__pred{i}") for i in range(len(preds)))
            return _Compiled(
                f"SELECT {_cols(node.schema)} FROM ({inner}) t WHERE {conds}",
                c.order)
        conds = " AND ".join(compile_expr(pr, ctx) for pr in node.predicates)
        return _Compiled(f"SELECT * FROM ({c.sql}) t WHERE {conds}", c.order)

    if isinstance(node, p.Mutate):
        c = compile_plan(node.child)
        sql = c.sql
        schema = dict(node.child.schema)
        from .expr import infer_dtype
        for name, e in node.exprs:
            ctx = _Ctx(schema, node.child.groups)
            compiled = compile_expr(e, ctx)
            # explicit projection: duckdb's binder rejects windows in
            # `* REPLACE (...)`, so spell the select list out
            if name in schema:
                items = ", ".join(
                    f"{compiled} AS {q(name)}" if k == name else q(k)
                    for k in schema)
            else:
                items = ", ".join([*(q(k) for k in schema),
                                   f"{compiled} AS {q(name)}"])
            if node.child.groups and contains_agg(e):
                # duckdb 1.5.x binder bug: a window over a subquery whose
                # FROM contains a CTE fails. The working shape is a
                # MATERIALIZED CTE at the head of the window's own SELECT.
                cte = _cte_name()
                sql = (f"WITH {q(cte)} AS MATERIALIZED ({sql}) "
                       f"SELECT {items} FROM {q(cte)}")
            else:
                sql = f"SELECT {items} FROM ({sql}) t"
            schema[name] = infer_dtype(e, schema)
        return _Compiled(sql, c.order)

    if isinstance(node, p.Select):
        c = compile_plan(node.child)
        return _Compiled(f"SELECT {_cols(node.schema)} FROM ({c.sql}) t", c.order)

    if isinstance(node, p.Rename):
        c = compile_plan(node.child)
        lookup = {old: new for new, old in node.mapping}
        items = ", ".join(
            f"{q(old)} AS {q(lookup[old])}" if old in lookup else q(old)
            for old in node.child.schema)
        return _Compiled(f"SELECT {items} FROM ({c.sql}) t", c.order)

    if isinstance(node, p.Arrange):
        c = compile_plan(node.child)
        ctx = _Ctx(node.child.schema, ())
        parts = []
        for k in node.keys:
            key_expr = k.operand if isinstance(k, Desc) else k
            direction = "DESC" if isinstance(k, Desc) else "ASC"
            parts.append(f"{compile_expr(key_expr, ctx)} {direction} NULLS LAST")
        # stable: tiebreak on input row order (S3)
        inner_sql = (f"SELECT *, row_number() OVER () AS {q('__rn')} "
                     f"FROM ({c.sql}) t")
        order = ", ".join(parts) + f", {q('__rn')} ASC"
        sql = f"SELECT {_cols(node.schema)} FROM ({inner_sql}) t"
        return _Compiled(sql, order)

    if isinstance(node, p.Distinct):
        c = compile_plan(node.child)
        # dplyr distinct keeps the FIRST occurrence in input order
        cols = _cols(node.schema)
        inner = f"SELECT *, row_number() OVER () AS {q('__rn')} FROM ({c.sql}) t"
        sql = (f"SELECT {cols} FROM ({inner}) t "
               f"GROUP BY {cols} ORDER BY min({q('__rn')})")
        return _Compiled(f"SELECT {cols} FROM ({sql}) t", None)

    if isinstance(node, p.Slice):
        c = compile_plan(node.child)
        groups = node.child.groups
        # apply any pending order inside the child so row numbers follow it
        # (duckdb preserves insertion order through these operators)
        base = f"{c.sql} ORDER BY {c.order}" if c.order else c.sql
        if node.kind == "sample":
            if groups:
                raise DpyrError("slice_sample() on grouped frames is not supported yet")
            seed = node.seed if node.seed is not None else 0
            sql = (f"SELECT {_cols(node.schema)} FROM "
                   f"(SELECT *, hash(row_number() OVER () + {seed}) AS {q('__h')} "
                   f"FROM ({base}) t) t ORDER BY {q('__h')} LIMIT {node.n}")
            return _Compiled(f"SELECT {_cols(node.schema)} FROM ({sql}) t", None)
        part = f"PARTITION BY {', '.join(q(g) for g in groups)}" if groups else ""
        inner = (f"SELECT *, row_number() OVER ({part}) AS {q('__rn2')}, "
                 f"count(*) OVER ({part}) AS {q('__cnt')} FROM ({base}) t")
        if node.kind == "head":
            cond = f"{q('__rn2')} <= {node.n}"
        else:
            cond = f"{q('__rn2')} > {q('__cnt')} - {node.n}"
        sql = f"SELECT {_cols(node.schema)} FROM ({inner}) t WHERE {cond}"
        return _Compiled(sql, None)

    if isinstance(node, p.GroupBy):
        return compile_plan(node.child)

    if isinstance(node, p.Summarize):
        c = compile_plan(node.child)
        ctx = _Ctx(node.child.schema, node.child.groups, in_agg=True)
        aggs = ", ".join(f"{compile_expr(e, ctx)} AS {q(name)}"
                         for name, e in node.aggs)
        keys = list(node.child.groups)
        if not keys:
            return _Compiled(f"SELECT {aggs} FROM ({c.sql}) t", None)
        key_sql = ", ".join(q(k) for k in keys)
        sql = f"SELECT {key_sql}, {aggs} FROM ({c.sql}) t GROUP BY {key_sql}"
        order = ", ".join(f"{q(k)} ASC NULLS LAST" for k in keys)  # S7
        return _Compiled(sql, order)

    if isinstance(node, p.Join):
        return _compile_join(node)

    if isinstance(node, p.PivotLonger):
        c = compile_plan(node.child)
        index = [k for k in node.child.schema if k not in node.cols]
        idx = ", ".join(q(k) for k in index)
        idx_prefix = f"{idx}, " if idx else ""
        target = node.schema[node.values_to]
        cast_t = SQL_DTYPE.get(target, "VARCHAR")
        selects = [
            (f"SELECT {idx_prefix}{sql_lit(col)} AS {q(node.names_to)}, "
             f"CAST({q(col)} AS {cast_t}) AS {q(node.values_to)} FROM ({c.sql}) t")
            for col in node.cols
        ]
        sql = " UNION ALL ".join(selects)
        return _Compiled(f"SELECT {_cols(node.schema)} FROM ({sql}) t", None)

    if isinstance(node, p.PivotWider):
        raise AssertionError("PivotWider is materialized by the frame layer")

    raise AssertionError(f"unhandled plan node {type(node).__name__}")


def _reverse_order(order: str) -> str:
    parts = []
    for piece in order.split(", "):
        if " DESC" in piece:
            parts.append(piece.replace(" DESC", " ASC"))
        elif " ASC" in piece:
            parts.append(piece.replace(" ASC", " DESC"))
        else:
            parts.append(piece + " DESC")
    return ", ".join(parts)


def _compile_join(node: p.Join) -> _Compiled:
    left = compile_plan(node.left)
    right = compile_plan(node.right)
    eq = " AND ".join(
        f"l.{q(k)} IS NOT DISTINCT FROM r.{q(k)}" if node.na_matches == "na"
        else f"l.{q(k)} = r.{q(k)}"
        for k in node.on)

    if node.how in ("semi", "anti"):
        neg = "NOT " if node.how == "anti" else ""
        sql = (f"SELECT * FROM ({left.sql}) l WHERE {neg}EXISTS "
               f"(SELECT 1 FROM ({right.sql}) r WHERE {eq})")
        return _Compiled(sql, left.order)

    overlap = (set(node.left.schema) & set(node.right.schema)) - set(node.on)
    sx, sy = node.suffix

    def sel(schema: dict[str, DType], alias: str, suffix: str,
            skip_keys: bool) -> list[str]:
        items = []
        for c in schema:
            if skip_keys and c in node.on:
                continue
            out_name = c + suffix if c in overlap else c
            items.append(f"{alias}.{q(c)} AS {q(out_name)}")
        return items

    if node.how == "full":
        keys = [f"COALESCE(l.{q(k)}, r.{q(k)}) AS {q(k)}" for k in node.on]
    else:
        side = "r" if node.how == "right" else "l"
        keys = [f"{side}.{q(k)} AS {q(k)}" for k in node.on]
    items = keys + sel(node.left.schema, "l", sx, True) \
                 + sel(node.right.schema, "r", sy, True)
    jt = {"inner": "INNER JOIN", "left": "LEFT JOIN", "right": "RIGHT JOIN",
          "full": "FULL JOIN"}[node.how]
    sql = (f"SELECT {', '.join(items)} FROM ({left.sql}) l {jt} "
           f"({right.sql}) r ON {eq}")
    ordered_cols = ", ".join(q(c) for c in node.schema)
    return _Compiled(f"SELECT {ordered_cols} FROM ({sql}) t", None)


def final_sql(node: p.PlanNode) -> str:
    c = compile_plan(node)
    return f"{c.sql} ORDER BY {c.order}" if c.order else c.sql


def execute(node: p.PlanNode) -> pl.DataFrame:
    import polars as pl

    from .polars_backend import PL_DTYPE, _normalize

    payload = resolve(sources_of(node)[0].token)
    assert isinstance(payload, DuckPayload)
    sql = final_sql(node)
    out = pl.from_arrow(payload.con.execute(sql).to_arrow_table())
    assert isinstance(out, pl.DataFrame)
    out = _normalize(out.lazy()).collect()
    # align dtypes with the plan schema (duckdb may widen/narrow)
    casts = [pl.col(k).cast(PL_DTYPE[v])
             for k, v in node.schema.items()
             if v in PL_DTYPE and out.schema.get(k) != PL_DTYPE[v]]
    if casts:
        out = out.with_columns(casts)
    return out.select(list(node.schema))
