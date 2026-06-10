"""Backend registry: maps Source tokens to executable payloads.

A plan is pure metadata; the data lives here. Each Source node's token
resolves to a payload describing where its rows come from. Mixing backends
in one plan (e.g. joining a polars frame to a duckdb table) is rejected
with a clear error rather than silently copying data.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .errors import DpyrError
from .plan import PlanNode, Source

if TYPE_CHECKING:
    import duckdb
    import polars as pl


class BackendError(DpyrError):
    pass


@dataclass
class PolarsPayload:
    lf: pl.LazyFrame
    kind: str = "polars"


@dataclass
class DuckPayload:
    con: duckdb.DuckDBPyConnection
    table_sql: str  # a quoted table/view name or a (subquery)
    kind: str = "duckdb"


_REGISTRY: dict[str, PolarsPayload | DuckPayload] = {}
_counter = itertools.count()


def register(payload: PolarsPayload | DuckPayload, hint: str = "mem") -> str:
    token = f"{hint}:{next(_counter)}"
    _REGISTRY[token] = payload
    return token


def resolve(token: str) -> PolarsPayload | DuckPayload:
    if token not in _REGISTRY:
        raise BackendError(
            f"source token '{token}' has no data attached. Frames built with "
            "from_schema() are schema-only; use from_polars/from_dict/"
            "read_parquet/read_csv/from_duckdb to attach data."
        )
    return _REGISTRY[token]


def sources_of(node: PlanNode) -> list[Source]:
    out: list[Source] = []

    def rec(n: PlanNode) -> None:
        if isinstance(n, Source):
            out.append(n)
            return
        for f in n.__dataclass_fields__:
            v = getattr(n, f)
            if isinstance(v, PlanNode):
                rec(v)

    rec(node)
    return out


def backend_kind(node: PlanNode) -> str:
    """The engine a plan runs on. In-memory (arrow/polars) frames bridge
    into duckdb zero-copy; tables on OTHER duckdb connections (a second
    .db file, a sqlite file, ...) bridge by streaming through arrow onto
    the plan's primary connection (S27)."""
    payloads = [resolve(s.token) for s in sources_of(node)]
    if not payloads:
        raise BackendError("plan has no sources")
    return ("duckdb" if any(isinstance(p, DuckPayload) for p in payloads)
            else "polars")


def payload_of(node: PlanNode) -> Any:
    return resolve(sources_of(node)[0].token)
