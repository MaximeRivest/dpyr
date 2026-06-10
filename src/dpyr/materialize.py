"""Materialization model (Epic 3): as lazy as possible internally, as eager
as possible observably.

- collect(plan) executes on whichever backend the plan's sources live on,
  returning a polars DataFrame (the interchange format).
- Results are cached by plan hash, so repeated notebook displays don't
  recompute (DESIGN §3 "repeated re-execution").
- persist(frame) collects and rebinds the frame to a new in-memory source
  on the same backend — the explicit checkpoint / snapshot operator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import plan as p
from .backend import DuckPayload, PolarsPayload, backend_kind, register, sources_of

if TYPE_CHECKING:
    import polars as pl


class Options:
    """Global behavior switches. `interactive=True` (default) auto-collects
    at display/export boundaries; set False (or use .lazy()) for pipelines
    that must never execute implicitly."""

    interactive: bool = True
    preview_rows: int = 10


options = Options()

_CACHE: dict[str, pl.DataFrame] = {}


def cache_clear() -> None:
    _CACHE.clear()


def cache_size() -> int:
    return len(_CACHE)


def collect(node: p.PlanNode, *, use_cache: bool = True) -> pl.DataFrame:
    key = p.plan_hash(node)
    if use_cache and key in _CACHE:
        return _CACHE[key]
    if isinstance(node, p.PivotWider):
        # schema needs data: pivot in polars on either backend (DESIGN §3)
        child = collect(node.child, use_cache=use_cache)
        index = [c for c in node.child.schema
                 if c not in (node.names_from, node.values_from)]
        out = child.pivot(on=node.names_from, values=node.values_from,
                          index=index, aggregate_function="first")
    else:
        kind = backend_kind(node)
        if kind == "polars":
            from .polars_backend import compile_plan
            out = compile_plan(node).collect()
        else:
            from .duckdb_backend import execute as duck_execute
            out = duck_execute(node)
    if use_cache:
        _CACHE[key] = out
    return out


def persist_source(node: p.PlanNode) -> p.Source:
    """Materialize a plan and return a Source bound to the result, staying
    on the original backend (duckdb gets a temp table, S/DESIGN §3)."""
    import polars as pl

    from .polars_backend import schema_from_polars

    kind = backend_kind(node)
    df = collect(node)
    schema = schema_from_polars(df.lazy())
    if kind == "duckdb":
        from .backend import resolve
        payload = resolve(sources_of(node)[0].token)
        assert isinstance(payload, DuckPayload)
        con = payload.con
        token = register(DuckPayload(con, ""), hint="duck-persist")
        tmp = f"dpyr_persist_{token.split(':')[1]}"
        arrow_tbl = df.to_arrow()  # noqa: F841  (registered by name below)
        con.register(f"{tmp}_arrow", arrow_tbl)
        con.execute(f'CREATE TEMP TABLE "{tmp}" AS SELECT * FROM "{tmp}_arrow"')
        con.unregister(f"{tmp}_arrow")
        from .backend import _REGISTRY
        _REGISTRY[token] = DuckPayload(con, f'"{tmp}"')
        return p.Source(tmp, tuple(schema.items()), token)
    token = register(PolarsPayload(pl.LazyFrame(df)), hint="persist")
    return p.Source("persisted", tuple(schema.items()), token)


def preview(node: p.PlanNode, n_rows: int) -> tuple[pl.DataFrame, int | None]:
    """A cheap head-of-result for repr: full cached result if available,
    else a limited collect. Returns (head, total_rows_or_None)."""
    key = p.plan_hash(node)
    if key in _CACHE:
        full = _CACHE[key]
        return full.head(n_rows), full.height
    kind = backend_kind(node)
    if kind == "polars":
        from .polars_backend import compile_plan
        lf = compile_plan(node)
        head = lf.limit(n_rows + 1).collect()
        if head.height <= n_rows:
            return head.head(n_rows), head.height
        return head.head(n_rows), None
    full = collect(node)
    return full.head(n_rows), full.height
