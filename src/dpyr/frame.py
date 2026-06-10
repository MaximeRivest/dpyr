"""DFrame / GroupedDFrame: the user-facing verb surface.

A DFrame is an immutable handle on a logical plan. Verbs return new frames;
schema validation happens inside the plan node constructors, so mistakes
raise here, on the user's line (schema-eager). Materialization is automatic
at display/export boundaries (display-eager) unless the frame is .lazy().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, Iterator, Literal, TypeVar, cast

import functools

from . import plan as p
from .backend import BackendError, DuckPayload, PolarsPayload, register
from .dtypes import DType
from .errors import ColumnNotFoundError, ExprTypeError, GroupError
from .expr import Col, Desc, Expr, typed_col
from .expr import desc as desc_
from .materialize import collect, options, persist_source, preview

if TYPE_CHECKING:
    import duckdb
    import pandas as pd
    import polars as pl

ColRef = str | Col
S = TypeVar("S", bound="ColsProxy")
IntoPredicate = Expr | Callable[["ColsProxy"], Expr]

_VERBS = ("filter", "mutate", "select", "rename", "arrange", "distinct",
          "slice_head", "slice_tail", "slice_sample", "group_by",
          "summarize", "summarise", "count", "inner_join", "left_join",
          "right_join", "full_join", "semi_join", "anti_join",
          "pivot_longer", "pivot_wider", "pull", "separate", "unite",
          "relocate", "slice_min", "slice_max")


def _polish_tracebacks(cls: type) -> type:
    """Re-raise dpyr errors from the verb call itself, so the user sees
    their line plus one frame instead of the schema engine internals."""
    from .errors import DpyrError

    def wrap(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def verb(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except DpyrError as err:
                raise err.with_traceback(None) from None
        return verb

    for name in _VERBS:
        fn = cls.__dict__.get(name)
        if fn is not None:
            setattr(cls, name, wrap(fn))
    return cls


def _name(ref: ColRef, what: str) -> str:
    if isinstance(ref, Col):
        return ref.name
    if isinstance(ref, str):
        return ref
    raise TypeError(f"{what} expects column names or col.<name>, got {type(ref).__name__}")


class ColsProxy:
    """Frame-bound column proxy: `df.c.height` is a typed column expression
    and tab-completes from the live schema (Epic 7)."""

    def __init__(self, schema: dict[str, DType]):
        object.__setattr__(self, "_schema", schema)

    def __getattr__(self, name: str) -> Col:
        schema: dict[str, DType] = object.__getattribute__(self, "_schema")
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in schema:
            raise ColumnNotFoundError(name, schema, "df.c")
        return typed_col(name, schema[name])

    def __getitem__(self, name: str) -> Col:
        return self.__getattr__(name)

    def __dir__(self) -> list[str]:
        return list(object.__getattribute__(self, "_schema"))


@_polish_tracebacks
class DFrame(Generic[S]):
    """An ungrouped tidy frame (lazy plan + eager schema).

    Generic over a ColsProxy schema class: annotate frames as
    `DFrame[MyCols]` (see `dpyr stubgen`) and lambda verbs get statically
    typed column access — `f.filter(lambda c: c.height > 180)` completes
    and type-checks."""

    _interactive: bool = True

    def __init__(self, plan_node: p.PlanNode) -> None:
        if plan_node.groups:
            raise GroupError("internal: DFrame built from a grouped plan")
        self._plan = plan_node

    # -- metadata ------------------------------------------------------
    @property
    def plan(self) -> p.PlanNode:
        return self._plan

    @property
    def schema(self) -> dict[str, DType]:
        return dict(self._plan.schema)

    @property
    def columns(self) -> list[str]:
        return list(self._plan.schema)

    @property
    def c(self) -> S:
        return cast("S", ColsProxy(dict(self._plan.schema)))

    def __dir__(self) -> list[str]:  # runtime completion (Epic 7)
        return [*super().__dir__(), *self._plan.schema]

    def _spawn(self, node: p.PlanNode) -> DFrame:
        # dispatch on grouping so every verb works on grouped frames too
        out: DFrame = GroupedDFrame(node) if node.groups else DFrame(node)
        out._interactive = self._interactive
        return out

    def _spawn_grouped(self, node: p.PlanNode) -> GroupedDFrame:
        out: GroupedDFrame = GroupedDFrame(node)
        out._interactive = self._interactive
        return out

    def _resolve_preds(self, predicates: tuple[IntoPredicate, ...]) -> tuple[Expr, ...]:
        return tuple(q(self.c) if callable(q) else q for q in predicates)

    def _resolve_exprs(self, args: tuple[Any, ...],
                       kwargs: dict[str, Any]) -> tuple[tuple[str, Expr], ...]:
        from .tidyselect import Across
        out: list[tuple[str, Expr]] = []
        # dplyr's across() never touches grouping variables
        non_group = {k: v for k, v in self._plan.schema.items()
                     if k not in self._plan.groups}
        for a in args:
            if not isinstance(a, Across):
                raise TypeError(
                    "positional arguments to mutate()/summarize() must be "
                    "across(...); name everything else: mutate(x=...)")
            out.extend(a.expand(non_group))
        for k, v in kwargs.items():
            out.append((k, v(self.c) if callable(v) else v))
        return tuple(out)

    # -- materialization (Epic 3) ---------------------------------------
    def collect(self) -> pl.DataFrame:
        """Execute the plan and return a polars DataFrame."""
        return collect(self._plan)

    def to_polars(self) -> pl.DataFrame:
        return self.collect()

    def to_pandas(self) -> pd.DataFrame:
        try:
            import pandas  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "to_pandas() needs pandas: pip install 'dpyr[pandas]'") from err
        return self.collect().to_pandas()

    def pull(self, column: ColRef | None = None) -> list[Any]:
        name = _name(column, "pull()") if column is not None else self.columns[-1]
        if name not in self._plan.schema:
            raise ColumnNotFoundError(name, self._plan.schema, "pull()")
        return self.collect()[name].to_list()

    def persist(self) -> DFrame:
        """Materialize now and rebind to the result (snapshot checkpoint).
        Grouping survives, like dplyr's compute()."""
        node: p.PlanNode = persist_source(self._plan)
        if self._plan.groups:
            node = p.GroupBy(node, self._plan.groups)
        return self._spawn(node)

    def lazy(self) -> DFrame:
        """A copy whose repr/len never trigger execution."""
        out = self._spawn(self._plan)
        out._interactive = False
        return out

    def eager(self) -> DFrame:
        out = self._spawn(self._plan)
        out._interactive = True
        return out

    def __len__(self) -> int:
        return self.collect().height

    @property
    def shape(self) -> tuple[int, int]:
        return len(self), len(self.columns)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.collect().iter_rows(named=True))

    def __getitem__(self, column: str) -> list[Any]:
        return self.pull(column)

    def _provenance(self) -> str:
        try:
            from .backend import backend_kind
            kind = backend_kind(self._plan)
        except BackendError:
            kind = "schema-only"
        return kind

    def __repr__(self) -> str:
        kind = self._provenance()
        head = f"# dpyr frame · source: {kind}"
        if kind == "schema-only" or not (options.interactive and self._interactive):
            cols = ", ".join(f"{k} <{v!r}>" for k, v in self._plan.schema.items())
            return f"{head} (lazy)\n# columns: {cols}"
        rows, total = preview(self._plan, options.preview_rows)
        shown = rows.height
        total_s = str(total) if total is not None else "?"
        body = repr(rows)
        return f"{head} · showing {shown} of {total_s} rows\n{body}"

    def _repr_html_(self) -> str | None:
        kind = self._provenance()
        if kind == "schema-only" or not (options.interactive and self._interactive):
            return None
        rows, total = preview(self._plan, options.preview_rows)
        total_s = str(total) if total is not None else "unknown"
        html = rows._repr_html_() or ""
        return (f"<div><small># dpyr · source: {kind} · showing "
                f"{rows.height} of {total_s} rows</small>{html}</div>")

    # -- verbs ----------------------------------------------------------
    def filter(self, *predicates: IntoPredicate) -> DFrame:
        return self._spawn(p.Filter(self._plan, self._resolve_preds(predicates)))

    def mutate(self, *args: Any, **exprs: Expr | Callable[[ColsProxy], Expr]) -> DFrame:
        return self._spawn(p.Mutate(self._plan, self._resolve_exprs(args, exprs)))

    def select(self, *cols: Any) -> DFrame:
        from .tidyselect import resolve_selection
        keep = resolve_selection(cols, self._plan.schema, "select()")
        return self._spawn(p.Select(self._plan, keep))

    def rename(self, **mapping: ColRef) -> DFrame:
        pairs = tuple((new, _name(old, "rename()")) for new, old in mapping.items())
        return self._spawn(p.Rename(self._plan, pairs))

    def arrange(self, *keys: Expr | Desc) -> DFrame:
        return self._spawn(p.Arrange(self._plan, keys))

    def distinct(self, *cols: ColRef) -> DFrame:
        return self._spawn(p.Distinct(self._plan, tuple(_name(c, "distinct()") for c in cols)))

    def slice_head(self, n: int = 5) -> DFrame:
        return self._spawn(p.Slice(self._plan, "head", n))

    def slice_tail(self, n: int = 5) -> DFrame:
        return self._spawn(p.Slice(self._plan, "tail", n))

    def slice_sample(self, n: int = 5, seed: int | None = None) -> DFrame:
        if seed is None:
            import random
            # a drawn seed enters the plan hash, so the cache can never
            # freeze a "random" sample, and both backends share the seed
            seed = random.getrandbits(31)
        return self._spawn(p.Slice(self._plan, "sample", n, seed))

    def group_by(self, *keys: ColRef) -> GroupedDFrame:
        return self._spawn_grouped(
            p.GroupBy(self._plan, tuple(_name(k, "group_by()") for k in keys)))

    def summarize(self, *args: Any, **aggs: Expr | Callable[[ColsProxy], Expr]) -> DFrame:
        return self._spawn(p.Summarize(self._plan, self._resolve_exprs(args, aggs)))

    summarise = summarize

    def count(self, *cols: ColRef, name: str = "n") -> DFrame:
        from .expr import n as n_
        if not cols:
            return self.summarize(**{name: n_()})
        # on a grouped frame the result stays grouped by the original keys,
        # exactly like dplyr's count()
        return self.group_by(*cols).summarize(**{name: n_()})

    # -- tidyr-style reshapers ------------------------------------------
    def separate(self, column: ColRef, into: list[str], sep: str = "_",
                 remove: bool = True) -> DFrame:
        return self._spawn(p.Separate(self._plan, _name(column, "separate()"),
                                      tuple(into), sep, remove))

    def unite(self, new: str, cols: list[ColRef], sep: str = "_",
              remove: bool = True, na_rm: bool = False) -> DFrame:
        names = tuple(_name(c, "unite()") for c in cols)
        return self._spawn(p.Unite(self._plan, new, names, sep, remove, na_rm))

    def relocate(self, *cols: Any, before: ColRef | None = None,
                 after: ColRef | None = None) -> DFrame:
        from .tidyselect import resolve_selection
        moved = list(resolve_selection(cols, self._plan.schema, "relocate()"))
        rest = [c for c in self._plan.schema if c not in moved]
        if before is not None and after is not None:
            raise ExprTypeError("relocate() takes before= or after=, not both")
        if before is not None:
            anchor = _name(before, "relocate()")
            i = rest.index(anchor)
            order = rest[:i] + moved + rest[i:]
        elif after is not None:
            anchor = _name(after, "relocate()")
            i = rest.index(anchor) + 1
            order = rest[:i] + moved + rest[i:]
        else:
            order = moved + rest  # dplyr default: move to the front
        return self._spawn(p.Select(self._plan, tuple(order)))

    def _slice_rank(self, order_by: Expr, n: int, with_ties: bool,
                    descending: bool) -> DFrame:
        from .expr import Window, min_rank, row_number  # noqa: F401
        key = desc_(order_by) if descending else order_by
        if with_ties:
            rank = min_rank(key)
            out = self.filter(rank <= n)
        else:
            arranged = self.arrange(key)
            return arranged.slice_head(n)
        return out.arrange(key)

    def slice_min(self, order_by: Expr, n: int = 1,
                  with_ties: bool = True) -> DFrame:
        """Rows with the n smallest values (ties kept by default, like dplyr)."""
        return self._slice_rank(order_by, n, with_ties, descending=False)

    def slice_max(self, order_by: Expr, n: int = 1,
                  with_ties: bool = True) -> DFrame:
        return self._slice_rank(order_by, n, with_ties, descending=True)

    # -- joins -----------------------------------------------------------
    def _join(self, other: DFrame, how: p.JoinHow, on: ColRef | list[ColRef],
              suffix: tuple[str, str], na_matches: Literal["na", "never"]) -> DFrame:
        if isinstance(other, GroupedDFrame):
            raise GroupError("joining a grouped frame is not supported; ungroup() it first")
        refs = on if isinstance(on, list) else [on]
        keys = tuple(_name(r, "join on=") for r in refs)
        return self._spawn(p.Join(self._plan, other._plan, how, keys, suffix, na_matches))

    def inner_join(self, other: DFrame, on: ColRef | list[ColRef],
                   suffix: tuple[str, str] = (".x", ".y"),
                   na_matches: Literal["na", "never"] = "na") -> DFrame:
        return self._join(other, "inner", on, suffix, na_matches)

    def left_join(self, other: DFrame, on: ColRef | list[ColRef],
                  suffix: tuple[str, str] = (".x", ".y"),
                  na_matches: Literal["na", "never"] = "na") -> DFrame:
        return self._join(other, "left", on, suffix, na_matches)

    def right_join(self, other: DFrame, on: ColRef | list[ColRef],
                   suffix: tuple[str, str] = (".x", ".y"),
                   na_matches: Literal["na", "never"] = "na") -> DFrame:
        return self._join(other, "right", on, suffix, na_matches)

    def full_join(self, other: DFrame, on: ColRef | list[ColRef],
                  suffix: tuple[str, str] = (".x", ".y"),
                  na_matches: Literal["na", "never"] = "na") -> DFrame:
        return self._join(other, "full", on, suffix, na_matches)

    def semi_join(self, other: DFrame, on: ColRef | list[ColRef]) -> DFrame:
        return self._join(other, "semi", on, (".x", ".y"), "na")

    def anti_join(self, other: DFrame, on: ColRef | list[ColRef]) -> DFrame:
        return self._join(other, "anti", on, (".x", ".y"), "na")

    # -- reshaping ---------------------------------------------------------
    def pivot_longer(self, cols: list[Any], names_to: str = "name",
                     values_to: str = "value") -> DFrame:
        from .tidyselect import resolve_selection
        keep = resolve_selection(tuple(cols), self._plan.schema, "pivot_longer()")
        return self._spawn(p.PivotLonger(self._plan, keep, names_to, values_to))

    def pivot_wider(self, names_from: ColRef, values_from: ColRef) -> DFrame:
        node = p.PivotWider(self._plan, _name(names_from, "pivot_wider()"),
                            _name(values_from, "pivot_wider()"))
        try:
            # schema needs data: implicitly persist (DESIGN §3)
            return self._spawn(persist_source(node))
        except BackendError:
            return self._spawn(node)  # schema-only frames stay symbolic


@_polish_tracebacks
class GroupedDFrame(DFrame[S]):
    """A grouped frame: its own type so completion and semantics differ."""

    def __init__(self, plan_node: p.PlanNode) -> None:
        if not plan_node.groups:
            raise GroupError("internal: GroupedDFrame built from an ungrouped plan")
        self._plan = plan_node

    @property
    def groups(self) -> tuple[str, ...]:
        return self._plan.groups

    def __repr__(self) -> str:
        base = super().__repr__()
        return base.replace("# dpyr frame", f"# dpyr frame · groups: {', '.join(self.groups)}")

    def ungroup(self) -> DFrame:
        out: DFrame = DFrame(p.Ungroup(self._plan))
        out._interactive = self._interactive
        return out

    def filter(self, *predicates: IntoPredicate) -> GroupedDFrame:
        return self._spawn_grouped(p.Filter(self._plan, self._resolve_preds(predicates)))

    def mutate(self, *args: Any, **exprs: Expr | Callable[[ColsProxy], Expr]) -> GroupedDFrame:
        return self._spawn_grouped(p.Mutate(self._plan, self._resolve_exprs(args, exprs)))

    def group_by(self, *keys: ColRef) -> GroupedDFrame:
        names = tuple(_name(k, "group_by()") for k in keys)
        return self._spawn_grouped(p.GroupBy(self._plan, self._plan.groups + names))

    def summarize(self, *args: Any, **aggs: Expr | Callable[[ColsProxy], Expr]) -> DFrame | GroupedDFrame:
        node = p.Summarize(self._plan, self._resolve_exprs(args, aggs))
        return self._spawn_grouped(node) if node.groups else self._spawn(node)

    summarise = summarize

    def arrange(self, *keys: Expr | Desc) -> GroupedDFrame:
        return self._spawn_grouped(p.Arrange(self._plan, keys))

    def select(self, *cols: Any) -> GroupedDFrame:
        from .tidyselect import resolve_selection
        keep = resolve_selection(cols, self._plan.schema, "select()")
        return self._spawn_grouped(p.Select(self._plan, keep))

    def slice_head(self, n: int = 5) -> GroupedDFrame:
        return self._spawn_grouped(p.Slice(self._plan, "head", n))

    def slice_tail(self, n: int = 5) -> GroupedDFrame:
        return self._spawn_grouped(p.Slice(self._plan, "tail", n))


# ---------------------------------------------------------------------
# sources


def from_schema(schema: dict[str, DType], name: str = "table") -> DFrame:
    """A schema-only frame: full validation, no data. Useful for plan/IR
    work and tests; attach data with the other constructors."""
    return DFrame(p.Source(name, tuple(schema.items())))


def from_polars(df: pl.DataFrame | pl.LazyFrame, name: str = "polars") -> DFrame:
    import polars as pl

    from .polars_backend import _normalize, schema_from_polars
    lf = _normalize(df.lazy() if isinstance(df, pl.DataFrame) else df)
    token = register(PolarsPayload(lf), hint="mem")
    return DFrame(p.Source(name, tuple(schema_from_polars(lf).items()), token))


def from_dict(data: dict[str, list[Any]], name: str = "data") -> DFrame:
    import polars as pl
    return from_polars(pl.DataFrame(data), name=name)


def from_pandas(df: pd.DataFrame, name: str = "pandas") -> DFrame:
    try:
        import polars as pl
        return from_polars(pl.from_pandas(df), name=name)
    except ImportError as err:
        raise ImportError(
            "from_pandas() needs pandas: pip install 'dpyr[pandas]'") from err


def _file_token(kind: str, path: str) -> str:
    # content-addressed: re-reading the same unchanged file shares a plan
    # hash (cache hits across notebook re-runs); edits change the token
    import os
    try:
        st = os.stat(path)
        return f"{kind}:{path}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return f"{kind}:{path}"


def read_parquet(path: str) -> DFrame:
    import polars as pl

    from .backend import _REGISTRY
    from .polars_backend import _normalize, schema_from_polars
    lf = _normalize(pl.scan_parquet(path))
    token = _file_token("parquet", path)
    _REGISTRY[token] = PolarsPayload(lf)
    return DFrame(p.Source(path, tuple(schema_from_polars(lf).items()), token))


def read_csv(path: str) -> DFrame:
    import polars as pl

    from .backend import _REGISTRY
    from .polars_backend import _normalize, schema_from_polars
    lf = _normalize(pl.scan_csv(path))
    token = _file_token("csv", path)
    _REGISTRY[token] = PolarsPayload(lf)
    return DFrame(p.Source(path, tuple(schema_from_polars(lf).items()), token))


def from_duckdb(con: duckdb.DuckDBPyConnection, table: str) -> DFrame:
    from .duckdb_backend import q, schema_from_duckdb
    quoted = q(table)
    schema = schema_from_duckdb(con, quoted)
    # stable per (connection, table): cell re-runs share the cache; after
    # mutating the table use dpyr.cache_clear() or .persist() (DESIGN §3)
    from .backend import _REGISTRY
    token = f"duck:{table}@{id(con)}"
    _REGISTRY[token] = DuckPayload(con, quoted)
    return DFrame(p.Source(table, tuple(schema.items()), token))


def read_sql(con: duckdb.DuckDBPyConnection, query: str, name: str = "sql") -> DFrame:
    from .duckdb_backend import schema_from_duckdb
    sub = f"({query})"
    schema = schema_from_duckdb(con, sub)
    token = register(DuckPayload(con, sub), hint="duck-sql")
    return DFrame(p.Source(name, tuple(schema.items()), token))
