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
from .backend import DuckPayload, PolarsPayload, backend_kind, register

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


def collect(node: p.PlanNode, *, use_cache: bool = True,
            engine: str | None = None) -> pl.DataFrame:
    key = p.plan_hash(node) + (f":{engine}" if engine else "")
    if use_cache and key in _CACHE:
        return _CACHE[key]
    if isinstance(node, p.PivotWider):
        # schema needs data: pivot in polars on either backend (DESIGN §3)
        import warnings

        import polars as pl
        child = collect(node.child, use_cache=use_cache)
        index = [c for c in node.child.schema
                 if c not in (node.names_from, node.values_from)]
        dup_keys = index + [node.names_from]
        n_dups = (child.group_by(dup_keys).len().filter(pl.col("len") > 1)
                  .height) if child.height else 0
        if n_dups:
            warnings.warn(
                f"pivot_wider(): values are not uniquely identified for "
                f"{n_dups} key combination(s); keeping the first value "
                "(dplyr would build list-columns)", stacklevel=3)
        if not index:  # dplyr returns a single row when there are no id cols
            child = child.with_columns(pl.lit(1).alias("__dpyr_idx"))
            out = child.pivot(on=node.names_from, values=node.values_from,
                              index=["__dpyr_idx"], aggregate_function="first"
                              ).drop("__dpyr_idx")
        else:
            out = child.pivot(on=node.names_from, values=node.values_from,
                              index=index, aggregate_function="first")
    else:
        kind = engine or backend_kind(node)
        if engine == "polars" and backend_kind(node) == "duckdb":
            from .backend import BackendError
            raise BackendError(
                "engine='polars' cannot read duckdb-resident tables; "
                "duckdb can read in-memory frames, not vice versa")
        if kind == "polars":
            from .polars_backend import compile_plan
            out = compile_plan(node).collect()
        else:
            from .duckdb_backend import execute as duck_execute
            out = duck_execute(node)
    if use_cache:
        if len(_CACHE) >= 256:  # bound memory in long sessions
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = out
    return out


def persist_source(node: p.PlanNode) -> p.Source:
    """Materialize a plan and return a Source bound to the result, staying
    on the original backend. On duckdb this is CREATE TEMP TABLE AS <sql>,
    executed entirely in the engine — the data never enters Python."""
    import polars as pl

    from .polars_backend import schema_from_polars

    kind = backend_kind(node)
    if kind == "duckdb" and not _plan_needs_python(node):
        from .backend import _REGISTRY
        from .duckdb_backend import (
            connection_of,
            final_sql,
            register_bridges,
            schema_from_duckdb,
        )
        con = connection_of(node)
        token = register(DuckPayload(con, ""), hint="duck-persist")
        tmp = f"dpyr_persist_{token.split(':')[1]}"
        bridged = register_bridges(con, node)
        try:
            con.execute(f'CREATE TEMP TABLE "{tmp}" AS {final_sql(node)}')
        finally:
            for name in bridged:
                con.unregister(name)
        _REGISTRY[token] = DuckPayload(con, f'"{tmp}"')
        schema = schema_from_duckdb(con, f'"{tmp}"')
        return p.Source(tmp, tuple(schema.items()), token)
    df = collect(node)
    schema = schema_from_polars(df.lazy())
    if kind == "duckdb":  # plans with python-side ops (pivot_wider)
        from .backend import _REGISTRY
        from .duckdb_backend import connection_of
        con = connection_of(node)
        token = register(DuckPayload(con, ""), hint="duck-persist")
        tmp = f"dpyr_persist_{token.split(':')[1]}"
        con.register(f"{tmp}_arrow", df.to_arrow())
        con.execute(f'CREATE TEMP TABLE "{tmp}" AS SELECT * FROM "{tmp}_arrow"')
        con.unregister(f"{tmp}_arrow")
        _REGISTRY[token] = DuckPayload(con, f'"{tmp}"')
        return p.Source(tmp, tuple(schema.items()), token)
    token = register(PolarsPayload(pl.LazyFrame(df)), hint="persist")
    return p.Source("persisted", tuple(schema.items()), token)


def _plan_needs_python(node: p.PlanNode) -> bool:
    """True if any node must materialize through polars (pivot_wider)."""
    if isinstance(node, p.PivotWider):
        return True
    for f in node.__dataclass_fields__:
        v = getattr(node, f)
        if isinstance(v, p.PlanNode) and _plan_needs_python(v):
            return True
    return False


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
