"""Row functions: any Python function, run once per row, as a column (S38).

    from dpyr import vectorize, col

    @vectorize
    def slug(title: str, sep: str = "-") -> str:
        return sep.join(title.lower().split())

    df.mutate(s=slug(col.title))                  # one call per row
    df.mutate(s=slug(col.title, sep="_"))         # constants are the same on every row
    df.filter(is_spam(col.subject, col.body))     # several columns, a bool result

Like numpy.vectorize, this is a convenience, not a speed-up: the function
runs in Python, once per distinct row of arguments. What dpyr adds:

- the column's type comes from the return annotation (or ``dtype=``), so the
  chain is still checked when you write it;
- identical arguments are computed once, and every result is remembered for
  the session: displaying a frame, then collecting it, then building a new
  chain on it never calls the function twice for the same arguments;
- a displayed frame runs the function only on the rows it shows, when the
  steps after it keep rows one-for-one;
- ``threads=`` runs rows concurrently (for functions that wait on the
  network, like model calls);
- ``errors="raise"`` (default) raises after every row has run, so a second
  run retries only the rows that failed; ``errors="null"`` keeps going.

The frame layer turns each call into a RowMap step (plan.py), placed before
the expression that uses it; materialize.run_row_maps runs those steps in
Python and hands the result back to the engine as an in-memory table.
"""

from __future__ import annotations

import collections.abc
import concurrent.futures
import contextvars
import dataclasses
import datetime as _dt
import enum
import hashlib
import json
import re
import types
import typing
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from . import plan as p
from .dtypes import DType
from .errors import DpyrError, ExprTypeError
from .expr import Col, Expr, RowCall, _children

# -- the column type -----------------------------------------------------------


def _pl_dtype(tp: Any, where: str) -> Any:
    """The polars dtype for a Python annotation."""
    import polars as pl
    if tp is str:
        return pl.String
    if tp is bool:
        return pl.Boolean
    if tp is int:
        return pl.Int64
    if tp is float:
        return pl.Float64
    if tp is _dt.datetime:
        return pl.Datetime("us")
    if tp is _dt.date:
        return pl.Date
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1:
            return _pl_dtype(rest[0], where)
        raise ExprTypeError(f"{where}: a column holds one type, not {tp}")
    if origin is typing.Literal:
        kinds = {type(a) for a in args}
        if len(kinds) == 1:
            return _pl_dtype(kinds.pop(), where)
        raise ExprTypeError(f"{where}: a column holds one type, not {tp}")
    if origin in (list, set, frozenset, collections.abc.Sequence, collections.abc.Set) or \
            (origin is tuple and len(args) == 2 and args[1] is Ellipsis):
        if not args:
            raise ExprTypeError(f"{where}: say what the list holds, e.g. list[str]")
        return pl.List(_pl_dtype(args[0], where))
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        kinds = {type(m.value) for m in tp}
        if len(kinds) == 1:
            return _pl_dtype(kinds.pop(), where)
        raise ExprTypeError(f"{where}: enum {tp.__name__} mixes value types")
    fields = _struct_fields(tp)
    if fields is not None:
        return pl.Struct([pl.Field(k, _pl_dtype(v, f"{where}.{k}")) for k, v in fields.items()])
    raise ExprTypeError(
        f"{where}: can't tell what type of column {tp!r} makes; use str, int, float, bool, "
        "date, datetime, an Enum, list[...], a dataclass / pydantic model / TypedDict, "
        "or pass dtype=")


def _struct_fields(tp: Any) -> dict[str, Any] | None:
    if not isinstance(tp, type):
        return None
    if dataclasses.is_dataclass(tp) or typing.is_typeddict(tp):
        return typing.get_type_hints(tp)
    model_fields = getattr(tp, "model_fields", None)          # pydantic v2
    if isinstance(model_fields, dict):
        return {k: f.annotation for k, f in model_fields.items()}
    return None


def _resolve_dtype(fn: Callable, dtype: Any, name: str) -> tuple[DType, Any]:
    """(dpyr dtype, canonical polars dtype) of the function's results."""
    import polars as pl

    from .polars_backend import PL_DTYPE, canonical_pl, dtype_from_polars
    if dtype is None:
        import inspect
        try:
            hints = typing.get_type_hints(inspect.unwrap(fn))   # wrapped callables: the original's
        except Exception:  # noqa: BLE001 — unresolvable annotations: same as none
            hints = {}
        if "return" not in hints:
            raise ExprTypeError(
                f"vectorize({name}): can't tell what type {name} returns; annotate it "
                f"(def {name}(...) -> str) or pass dtype=str")
        dtype = hints["return"]
    if isinstance(dtype, DType):
        if dtype.nested or dtype not in PL_DTYPE:
            raise ExprTypeError(f"vectorize({name}): for a {dtype!r} column pass the "
                                "Python type instead, e.g. dtype=list[str]")
        pld = PL_DTYPE[dtype]
    elif isinstance(dtype, pl.DataType) or (isinstance(dtype, type) and issubclass(dtype, pl.DataType)):
        pld = dtype
    else:
        pld = _pl_dtype(dtype, f"vectorize({name})")
    canon = canonical_pl(pld)
    if canon is None:
        raise ExprTypeError(f"vectorize({name}): dpyr can't hold {pld} columns")
    return dtype_from_polars(name, canon), canon


# -- the function ------------------------------------------------------------------

# results of every row function this session: (id(fn), version) -> (fn, {args key: value})
_MEMO: dict[tuple[int, str], tuple[Callable, dict[str, Any]]] = {}


def memo_clear() -> None:
    _MEMO.clear()


class RowFunction:
    """A Python function made callable on columns; see ``vectorize``."""

    def __init__(self, fn: Callable, *, dtype: Any = None, threads: int | None = None,
                 errors: str = "raise", version: str = "", name: str | None = None) -> None:
        if errors not in ("raise", "null"):
            raise ValueError("errors must be 'raise' or 'null'")
        self.fn = fn
        self.name = name or getattr(fn, "__name__", None) or type(fn).__name__
        self.dtype, self.pl_dtype = _resolve_dtype(fn, dtype, self.name)
        self.threads = max(1, int(threads or 1))
        self.errors = errors
        self.version = version
        # identity (not just the name) and version take part in plan hashes
        self.label = f"{self.name}@{id(fn):x}" + (f"#{version}" if version else "")
        self.__name__ = self.name
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__wrapped__ = fn

    def __repr__(self) -> str:
        return f"<vectorized {self.name} -> {self.dtype!r}>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not any(isinstance(a, Expr) for a in (*args, *kwargs.values())):
            return self.fn(*args, **kwargs)            # plain values: a plain call
        return RowCall(self, tuple(args), tuple(kwargs.items()))

    def _memo(self) -> dict[str, Any]:
        key = (id(self.fn), self.version)
        if key not in _MEMO:
            _MEMO[key] = (self.fn, {})
        return _MEMO[key][1]


def vectorize(fn: Callable | None = None, *, dtype: Any = None, threads: int | None = None,
              errors: str = "raise", version: str = "") -> Any:
    """Make a Python function usable on columns: called with a column
    expression anywhere in its arguments, it returns a column expression for
    ``mutate()`` / ``filter()``; called with plain values, it just runs.

    - ``dtype``: the result type (default: the return annotation); a Python
      type (``str``, ``list[str]``, a dataclass, ...) or a dpyr dtype
    - ``threads``: rows run concurrently (for functions that wait on I/O);
      default 1, or the object's own default (see below)
    - ``errors``: ``"raise"`` (after every row ran; results that worked are
      kept, so running again retries only the failures) or ``"null"``
    - ``version``: change it when the function's behavior changes without
      its identity changing, so remembered results are not reused

    Use as ``vectorize(fn)``, ``@vectorize`` or ``@vectorize(dtype=...)``.

    An object can decide how it is vectorized by defining
    ``__dpyr_vectorize__(dtype=, threads=, errors=, version=)`` returning a
    RowFunction (an AI-function library does, to pin the prompt a column
    was computed with)."""
    def wrap(f: Callable) -> RowFunction:
        hook = getattr(f, "__dpyr_vectorize__", None)
        if callable(hook) and not isinstance(f, type):
            options = {"dtype": dtype, "threads": threads, "errors": errors, "version": version}
            return hook(**options)
        return RowFunction(f, dtype=dtype, threads=threads, errors=errors, version=version)
    return wrap if fn is None else wrap(fn)


# -- lifting row calls out of expressions ---------------------------------------------


def contains_row_call(e: Any) -> bool:
    if isinstance(e, RowCall):
        return True
    return isinstance(e, Expr) and any(contains_row_call(c) for c in _children(e))


def _rebuild(e: Any, f: Callable[[Any], Any]) -> Any:
    """``e`` with ``f`` applied to every direct sub-expression."""
    if isinstance(e, tuple):
        return tuple(_rebuild(x, f) if isinstance(x, tuple) else (f(x) if isinstance(x, Expr) else x)
                     for x in e)
    if not isinstance(e, Expr) or not dataclasses.is_dataclass(e):
        return e
    changes = {}
    for fld in dataclasses.fields(e):
        v = getattr(e, fld.name)
        if isinstance(v, Expr):
            changes[fld.name] = f(v)
        elif isinstance(v, tuple):
            changes[fld.name] = _rebuild(v, f)
    return dataclasses.replace(e, **changes) if changes else e


def _temp(kind: str, e: Expr) -> str:
    return f"__dpyr_{kind}_{hashlib.sha256(repr(e).encode()).hexdigest()[:10]}"


def _shown(e: Any) -> str:
    """How the user wrote an expression (a row call by its function's name)."""
    if isinstance(e, RowCall):
        parts = [_shown(a) for a in e.args] + [f"{k}={_shown(v)}" for k, v in e.kwargs]
        return f"{e.func.name}({', '.join(parts)})"
    if isinstance(e, Expr) and contains_row_call(e):
        return repr(_rebuild(e, lambda x: Col(_shown(x)) if isinstance(x, RowCall) else x))
    return repr(e)


_TEMP = re.compile(r"(col\.)?(__dpyr_(?:row|arg)_[0-9a-f]{10})")


def _user_facing(err: DpyrError, shown: dict[str, str]) -> DpyrError:
    """The error with helper column names replaced by what the user wrote."""
    msg = str(err)
    msg = _TEMP.sub(lambda m: shown.get(m.group(2), m.group(0)) if m.group(1) else "", msg)
    msg = re.sub(r"(, ){2,}", ", ", msg).replace(", ,", ",").replace(": , ", ": ").rstrip(", ")
    err.args = (msg, *err.args[1:])
    return err


def _lift(node: p.PlanNode, e: Any, temps: list[str],
          shown: dict[str, str]) -> tuple[p.PlanNode, Any]:
    """Move every row call in ``e`` into RowMap steps under ``node``;
    returns the new node and ``e`` with those calls replaced by columns."""
    if not isinstance(e, Expr) or not contains_row_call(e):
        return node, e
    if not isinstance(e, RowCall):
        box = [node]

        def sub(x: Any) -> Any:
            box[0], out = _lift(box[0], x, temps, shown)
            return out
        rebuilt = _rebuild(e, sub)
        return box[0], rebuilt
    spec: list[tuple[str | None, bool, Any]] = []
    for key, value in [(None, a) for a in e.args] + list(e.kwargs):
        if not isinstance(value, Expr):
            spec.append((key, False, value))
            continue
        original = value
        node, value = _lift(node, value, temps, shown)
        if not isinstance(value, Col):                 # compute the argument in the engine first
            name = _temp("arg", value)
            shown[name] = _shown(original)
            if name not in node.schema:
                node = p.Mutate(node, ((name, value),))
                temps.append(name)
            value = Col(name)
        spec.append((key, True, value.name))
    out = _temp("row", e)
    shown[out] = _shown(e)
    if out not in node.schema:
        node = p.RowMap(node, out, e.func, tuple(spec))
        temps.append(out)
    return node, Col(out)



def _drop(node: p.PlanNode, temps: list[str]) -> p.PlanNode:
    keep = tuple(k for k in node.schema if k not in temps)
    return p.Select(node, keep) if len(keep) != len(node.schema) else node


def lift_mutate(node: p.PlanNode, exprs: tuple[tuple[str, Expr], ...]) -> p.PlanNode:
    """``mutate(**exprs)`` with row calls: plain expressions stay together;
    each expression with a row call gets its RowMap steps first. Left-to-
    right order (later expressions see earlier ones) is kept."""
    temps: list[str] = []
    shown: dict[str, str] = {}
    pending: list[tuple[str, Expr]] = []
    try:
        for name, e in exprs:
            if contains_row_call(e):
                if pending:
                    node, pending = p.Mutate(node, tuple(pending)), []
                node, e = _lift(node, e, temps, shown)
            pending.append((name, e))
        if pending:
            node = p.Mutate(node, tuple(pending))
    except DpyrError as err:
        raise _user_facing(err, shown) from None
    return _drop(node, temps)


def lift_filter(node: p.PlanNode, predicates: tuple[Expr, ...]) -> p.PlanNode:
    temps: list[str] = []
    shown: dict[str, str] = {}
    lifted = []
    try:
        for e in predicates:
            node, e = _lift(node, e, temps, shown)
            lifted.append(e)
        node = p.Filter(node, tuple(lifted))
    except DpyrError as err:
        raise _user_facing(err, shown) from None
    return _drop(node, temps)


# -- running -------------------------------------------------------------------------


def _cell(v: Any) -> Any:
    if isinstance(v, enum.Enum):
        return _cell(v.value)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: _cell(getattr(v, f.name)) for f in dataclasses.fields(v)}
    dump = getattr(v, "model_dump", None)
    if callable(dump) and not isinstance(v, type):
        return _cell(dump())
    if isinstance(v, Mapping):
        return {k: _cell(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_cell(x) for x in v]
    return v


def _key(args: list[Any], kwargs: dict[str, Any]) -> str:
    return json.dumps([args, kwargs], sort_keys=True, default=repr)


def run(node: p.RowMap, df: Any) -> Any:
    """The RowMap's column (a polars Series) for the rows of ``df``."""
    import polars as pl
    func: RowFunction = node.func
    columns = {v: df[v].to_list() for _k, is_col, v in node.args if is_col}
    calls: list[tuple[list[Any], dict[str, Any]]] = []
    for i in range(df.height):
        args = [columns[v][i] if is_col else v for k, is_col, v in node.args if k is None]
        kwargs = {k: (columns[v][i] if is_col else v) for k, is_col, v in node.args if k is not None}
        calls.append((args, kwargs))
    keys = [_key(a, kw) for a, kw in calls]
    memo = func._memo()
    todo: dict[str, tuple[list[Any], dict[str, Any]]] = {}
    for k, c in zip(keys, calls):
        if k not in memo and k not in todo:
            todo[k] = c

    def one(item: tuple[str, tuple[list[Any], dict[str, Any]]]) -> tuple[str, bool, Any]:
        k, (a, kw) = item
        try:
            return k, True, _cell(func.fn(*a, **kw))
        except Exception as exc:  # noqa: BLE001 — reported for the rows that hit it
            return k, False, f"{type(exc).__name__}: {exc}"

    items = list(todo.items())
    if func.threads > 1 and len(items) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=func.threads) as pool:
            futures = [pool.submit(contextvars.copy_context().run, one, it) for it in items]
            results = [f.result() for f in futures]
    else:
        results = [one(it) for it in items]
    failed: dict[str, str] = {}
    for k, ok, value in results:
        if ok:
            memo[k] = value
        else:
            failed[k] = value
    if failed:
        rows = [i for i, k in enumerate(keys) if k in failed]
        first = rows[0]
        msg = (f"{func.name}() failed on {len(rows)} of {len(keys)} rows (row {first}: "
               f"{failed[keys[first]]})")
        if func.errors == "raise":
            raise DpyrError(msg + ". The results that worked are kept, so running again "
                                  "retries only these rows; vectorize(..., errors='null') "
                                  "leaves them missing instead")
        warnings.warn(msg + "; those rows are missing", stacklevel=2)
    values = [memo.get(k) for k in keys]
    try:
        return pl.Series(node.name, values, dtype=func.pl_dtype, strict=True)
    except Exception:  # noqa: BLE001 — find the value that does not fit
        for i, v in enumerate(values):
            try:
                pl.Series("x", [v], dtype=func.pl_dtype, strict=True)
            except Exception:  # noqa: BLE001
                raise DpyrError(
                    f"{func.name}() returned {type(v).__name__} {v!r:.80} on row {i}, which "
                    f"doesn't fit its {func.dtype!r} column; fix the function or its "
                    "return annotation (or pass dtype=)") from None
        raise


def attach(node: p.RowMap, df: Any) -> Any:
    """``df`` with the RowMap's column, in the RowMap's schema order."""
    col = run(node, df)
    return df.with_columns(col).select(list(node.schema))


__all__ = ["vectorize", "RowFunction", "lift_mutate", "lift_filter", "contains_row_call",
           "memo_clear"]
