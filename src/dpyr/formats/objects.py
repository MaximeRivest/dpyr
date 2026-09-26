"""In-memory object ingest: each block registers one kind of tabular
object that read() accepts. Predicates sniff module names so none of
these imports are required dependencies."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import numbers
from collections.abc import Mapping
from typing import Any

from ..errors import DpyrError
from .registry import object_reader, type_name


def _read_array(arr: Any):
    """1-D -> one 'value' column; 2-D -> column_0..column_n."""
    import polars as pl

    from ..frame import from_polars
    if arr.ndim == 1:
        return from_polars(pl.DataFrame({"value": arr}))
    if arr.ndim == 2:
        data = {f"column_{i}": arr[:, i] for i in range(arr.shape[1])}
        return from_polars(pl.DataFrame(data))
    raise DpyrError(
        f"read() takes 1-D or 2-D arrays, got {arr.ndim}-D shape "
        f"{tuple(arr.shape)}")


def _is_plain_dict(s: Any) -> bool:
    # Hugging Face DatasetDict subclasses dict; its reader runs first
    return isinstance(s, dict) and not type_name(s).startswith("datasets.")


def _read_dict(source: Any, table: Any):
    from ..frame import from_dict
    if table is not None:
        raise DpyrError("read(table=...) only applies to database sources "
                        "and Hugging Face dataset splits")
    return from_dict(source)


def _record(item: Any, i: int) -> Mapping[str, Any]:
    """One row: a dict (any Mapping), a dataclass, a namedtuple, or a
    pydantic model."""
    if isinstance(item, Mapping):
        return item
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return dataclasses.asdict(item)
    if isinstance(item, tuple) and hasattr(item, "_asdict"):
        return item._asdict()
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump()
    raise DpyrError(
        f"read() takes a list of records, one per row (dicts, dataclasses, "
        f"namedtuples or pydantic models); item {i} is of type "
        f"{type(item).__name__}. For one column of values use "
        "read({'value': [...]})")


def _kind(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, numbers.Integral):
        return "int"
    if isinstance(v, numbers.Real):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, _dt.datetime):
        return "datetime"
    if isinstance(v, _dt.date):
        return "date"
    if isinstance(v, (list, tuple)):
        return "list"
    if isinstance(v, Mapping):
        return "dict"
    return type(v).__name__


def _record_column(name: str, values: list[Any]) -> Any:
    """One column as an arrow array. Arrow infers over every value (polars
    samples the first rows and can silently truncate later ones), and the
    two mixes vctrs allows are combined first: bool with numbers counts as
    0/1, date with datetime becomes datetime."""
    import pyarrow as pa
    kinds: dict[str, int] = {}
    for i, v in enumerate(values):
        if v is not None:
            kinds.setdefault(_kind(v), i)
    known = {"bool", "int", "float", "str", "datetime", "date", "list", "dict"}
    for k, i in kinds.items():
        if k not in known:
            raise DpyrError(
                f"column '{name}' holds a {k} in row {i}, which a table can't "
                "store; convert it to numbers, strings, bools, dates, or "
                "lists/dicts of those first")
    if len(kinds) > 1 and "bool" in kinds and set(kinds) <= {"bool", "int", "float"}:
        values = [int(v) if isinstance(v, bool) else v for v in values]
    if set(kinds) == {"date", "datetime"}:
        values = [_dt.datetime.combine(v, _dt.time())
                  if _kind(v) == "date" else v for v in values]
    if len(kinds) > 1 and not set(kinds) <= {"bool", "int", "float"} \
            and set(kinds) != {"date", "datetime"}:
        (k1, i1), (k2, i2) = sorted(kinds.items(), key=lambda kv: kv[1])[:2]
        raise DpyrError(
            f"column '{name}' mixes {k1} (row {i1}) and {k2} (row {i2}) "
            "values; read() keeps one type per column, like "
            "dplyr::bind_rows()")
    try:
        return pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError) as err:
        raise DpyrError(f"column '{name}' can't be stored: {err}") from None


def _read_records(source: Any, table: Any):
    import polars as pl
    import pyarrow as pa

    from ..frame import from_polars
    if table is not None:
        raise DpyrError("read(table=...) only applies to database sources "
                        "and Hugging Face dataset splits")
    if not source:
        raise DpyrError(
            "read([]) has no rows to take column names from; for an empty "
            "table pass empty columns: read({'x': []})")
    rows = [_record(item, i) for i, item in enumerate(source)]
    # bind_rows semantics: the union of keys in first-seen order, a key a
    # row lacks is missing (null) in that row
    names = list(dict.fromkeys(k for r in rows for k in r))
    bad = [k for k in names if not isinstance(k, str)]
    if bad:
        raise DpyrError(f"record keys become column names and must be "
                        f"strings; got {bad[0]!r}")
    arrow = pa.table({k: _record_column(k, [r.get(k) for r in rows])
                      for k in names})
    out = pl.from_arrow(arrow)
    assert isinstance(out, pl.DataFrame)
    return from_polars(out, name="records")


def _is_duck_con(s: Any) -> bool:
    import duckdb
    return isinstance(s, duckdb.DuckDBPyConnection)


def _read_duck_con(source: Any, table: Any):
    from ..io import Database
    db = Database(source, "connection")
    return db.table(table) if table is not None else db


def _is_hf_dict(s: Any) -> bool:
    return (type_name(s).startswith("datasets.")
            and "Dict" in type(s).__name__)


def _read_hf_dict(source: Any, table: Any):
    from ..io import read
    splits = list(source.keys())
    if table is None:
        raise DpyrError(
            f"this Hugging Face dataset has splits {splits}; pick one: "
            f"read(ds, {splits[0]!r})")
    if table not in splits:
        from ..errors import ColumnNotFoundError
        raise ColumnNotFoundError(table, splits, "dataset splits")
    return read(source[table])


def _is_hf_dataset(s: Any) -> bool:
    return type_name(s).startswith("datasets.")


def _read_hf_dataset(source: Any, table: Any):
    import polars as pl

    from ..frame import from_polars
    arrow = source.data
    arrow = getattr(arrow, "table", arrow)  # unwrap datasets.table.Table
    out = pl.from_arrow(arrow)
    assert isinstance(out, pl.DataFrame)
    return from_polars(out)


def _is_polars(s: Any) -> bool:
    import polars as pl
    return isinstance(s, (pl.DataFrame, pl.LazyFrame))


def _read_polars(source: Any, table: Any):
    from ..frame import from_polars
    return from_polars(source)


def _read_pandas(source: Any, table: Any):
    import polars as pl

    from ..frame import from_polars
    return from_polars(pl.from_pandas(source))


def _read_arrow(source: Any, table: Any):
    import polars as pl

    from ..frame import from_polars
    out = pl.from_arrow(source)
    assert isinstance(out, pl.DataFrame)
    return from_polars(out)


def _read_numpy(source: Any, table: Any):
    return _read_array(source)


def _read_torch(source: Any, table: Any):
    return _read_array(source.detach().cpu().numpy())


def _read_jax(source: Any, table: Any):
    import numpy as np
    return _read_array(np.asarray(source))


# registration order matters only where predicates overlap: the HF
# DatasetDict (a dict subclass) must beat the plain-dict reader
object_reader("hf-splits", _is_hf_dict, _read_hf_dict)
object_reader("dict", _is_plain_dict, _read_dict)
object_reader("records", lambda s: isinstance(s, (list, tuple)), _read_records)
object_reader("duckdb-connection", _is_duck_con, _read_duck_con)
object_reader("polars", _is_polars, _read_polars)
object_reader("pandas", lambda s: type_name(s).startswith("pandas."), _read_pandas)
object_reader("arrow", lambda s: type_name(s).startswith("pyarrow."), _read_arrow)
object_reader("hf-dataset", _is_hf_dataset, _read_hf_dataset)
object_reader("numpy", lambda s: type_name(s).startswith("numpy."), _read_numpy)
object_reader("torch", lambda s: type_name(s).startswith("torch."), _read_torch)
object_reader("jax", lambda s: type_name(s).startswith(("jax.", "jaxlib.")), _read_jax)
