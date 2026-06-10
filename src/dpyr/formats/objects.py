"""In-memory object ingest: each block registers one kind of tabular
object that read() accepts. Predicates sniff module names so none of
these imports are required dependencies."""

from __future__ import annotations

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
object_reader("duckdb-connection", _is_duck_con, _read_duck_con)
object_reader("polars", _is_polars, _read_polars)
object_reader("pandas", lambda s: type_name(s).startswith("pandas."), _read_pandas)
object_reader("arrow", lambda s: type_name(s).startswith("pyarrow."), _read_arrow)
object_reader("hf-dataset", _is_hf_dataset, _read_hf_dataset)
object_reader("numpy", lambda s: type_name(s).startswith("numpy."), _read_numpy)
object_reader("torch", lambda s: type_name(s).startswith("torch."), _read_torch)
object_reader("jax", lambda s: type_name(s).startswith(("jax.", "jaxlib.")), _read_jax)
