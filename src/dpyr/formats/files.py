"""File formats: each block below is one independent format — a reader,
optionally a writer, and the suffixes it claims. Engine-side fast paths
(duckdb in-engine COPY, polars sinks) live here too, shared via helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import DpyrError
from .registry import file_format

if TYPE_CHECKING:
    from ..frame import DFrame


# -- shared write helpers -----------------------------------------------------

def _copy_in_engine(frame: DFrame, path: str, format_clause: str) -> bool:
    """duckdb plans write via in-engine COPY; returns False when the frame
    isn't duckdb-backed (caller falls back to polars)."""
    from ..backend import backend_kind
    from ..materialize import _plan_needs_python
    if backend_kind(frame.plan) != "duckdb" or _plan_needs_python(frame.plan):
        return False
    from ..duckdb_backend import connection_of, final_sql, register_bridges
    con = connection_of(frame.plan)
    escaped = path.replace("'", "''")
    bridged = register_bridges(con, frame.plan)
    try:
        con.execute(f"COPY ({final_sql(frame.plan)}) TO '{escaped}' "
                    f"({format_clause})")
    finally:
        for b in bridged:
            con.unregister(b)
    return True


def _sink_or_collect(frame: DFrame, path: str, sink: str, write: str,
                     **kwargs) -> None:
    """polars plans stream via sink_* when the plan supports it, else
    collect and write eagerly."""
    try:
        from ..polars_backend import compile_plan
        getattr(compile_plan(frame.plan), sink)(path, **kwargs)
    except Exception:
        getattr(frame.collect(), write)(path, **kwargs)


def _scan_source(name: str, lf, path: str) -> DFrame:
    from .. import plan as p
    from ..backend import _REGISTRY, PolarsPayload
    from ..frame import DFrame, _file_token
    from ..polars_backend import _normalize, schema_from_polars
    lf = _normalize(lf)
    token = _file_token(name, path)
    _REGISTRY[token] = PolarsPayload(lf)
    return DFrame(p.Source(path, tuple(schema_from_polars(lf).items()), token))


# -- parquet -------------------------------------------------------------------

def _read_parquet(path: str, table: str | None) -> DFrame:
    import polars as pl
    return _scan_source("parquet", pl.scan_parquet(path), path)


def _write_parquet(frame: DFrame, path: str, table: str | None) -> None:
    if not _copy_in_engine(frame, path, "FORMAT PARQUET"):
        _sink_or_collect(frame, path, "sink_parquet", "write_parquet")


file_format("parquet", (".parquet", ".pq"), _read_parquet, _write_parquet)


# -- csv / tsv (compressed variants read eagerly: scans can't gunzip) -----------

def _read_csv(path: str, table: str | None) -> DFrame:
    import polars as pl
    return _scan_source("csv", pl.scan_csv(path), path)


def _read_csv_gz(path: str, table: str | None) -> DFrame:
    import polars as pl
    return _scan_source("csv", pl.read_csv(path).lazy(), path)


def _read_tsv(path: str, table: str | None) -> DFrame:
    import polars as pl
    if path.lower().endswith(".gz"):
        lf = pl.read_csv(path, separator="\t").lazy()
    else:
        lf = pl.scan_csv(path, separator="\t")
    return _scan_source("tsv", lf, path)


def _write_csv(frame: DFrame, path: str, table: str | None) -> None:
    if not _copy_in_engine(frame, path, "FORMAT CSV, HEADER"):
        _sink_or_collect(frame, path, "sink_csv", "write_csv")


def _write_tsv(frame: DFrame, path: str, table: str | None) -> None:
    if not _copy_in_engine(frame, path, "FORMAT CSV, HEADER, DELIMITER '\t'"):
        _sink_or_collect(frame, path, "sink_csv", "write_csv", separator="\t")


file_format("csv", (".csv",), _read_csv, _write_csv)
file_format("csv.gz", (".csv.gz",), _read_csv_gz, None)
file_format("tsv", (".tsv",), _read_tsv, _write_tsv)
file_format("tsv.gz", (".tsv.gz",), _read_tsv, None)


# -- json: .json is a document, .jsonl/.ndjson are scannable lines ---------------

def _read_ndjson(path: str, table: str | None) -> DFrame:
    import polars as pl
    return _scan_source("ndjson", pl.scan_ndjson(path), path)


def _read_json(path: str, table: str | None) -> DFrame:
    import polars as pl
    return _scan_source("json", pl.read_json(path).lazy(), path)


def _write_ndjson(frame: DFrame, path: str, table: str | None) -> None:
    if not _copy_in_engine(frame, path, "FORMAT JSON"):
        _sink_or_collect(frame, path, "sink_ndjson", "write_ndjson")


def _write_json(frame: DFrame, path: str, table: str | None) -> None:
    frame.collect().write_json(path)


file_format("ndjson", (".jsonl", ".ndjson"), _read_ndjson, _write_ndjson)
file_format("json", (".json",), _read_json, _write_json)


# -- arrow IPC (memory-mapped on read) -------------------------------------------

def _read_ipc(path: str, table: str | None) -> DFrame:
    import polars as pl
    return _scan_source("ipc", pl.scan_ipc(path, memory_map=True), path)


def _write_ipc(frame: DFrame, path: str, table: str | None) -> None:
    _sink_or_collect(frame, path, "sink_ipc", "write_ipc")


file_format("arrow", (".arrow", ".feather", ".ipc"), _read_ipc, _write_ipc)


# -- excel (optional extra: dpyr[excel]) -------------------------------------------

def _read_xlsx(path: str, table: str | None) -> DFrame:
    import polars as pl
    try:
        df = pl.read_excel(path, sheet_name=table)
    except (ImportError, ModuleNotFoundError) as err:
        raise DpyrError(
            "reading .xlsx needs the excel extra: pip install 'dpyr[excel]'"
        ) from err
    return _scan_source("xlsx", df.lazy(), path)


def _write_xlsx(frame: DFrame, path: str, table: str | None) -> None:
    try:
        frame.collect().write_excel(path, worksheet=table or "Sheet1")
    except (ImportError, ModuleNotFoundError) as err:
        raise DpyrError(
            "writing .xlsx needs the excel extra: pip install 'dpyr[excel]'"
        ) from err


file_format("excel", (".xlsx",), _read_xlsx, _write_xlsx)


# -- duckdb database files ------------------------------------------------------------

def _read_duckdb_file(path: str, table: str | None):
    from ..io import read_duckdb
    return read_duckdb(path, table)


def _write_duckdb_file(frame: DFrame, path: str, table: str | None) -> DFrame:
    if table is None:
        raise DpyrError(
            f"write({path!r}) needs a table name: write({path!r}, 'orders')")
    return frame.write_duckdb(path, table)


file_format("duckdb", (".db", ".duckdb", ".ddb"),
            _read_duckdb_file, _write_duckdb_file, needs_table=True)


# -- sqlite (read-only, through duckdb's sqlite scanner) -----------------------------

def _read_sqlite(path: str, table: str | None):
    import duckdb

    from ..io import Database
    con = duckdb.connect()
    try:
        con.execute("INSTALL sqlite; LOAD sqlite;")
        escaped = path.replace("'", "''")
        con.execute(f"ATTACH '{escaped}' AS sqlite_db (TYPE SQLITE)")
        con.execute("USE sqlite_db")
    except duckdb.Error as err:
        raise DpyrError(
            f"could not open {path!r} as sqlite via duckdb's sqlite "
            f"extension: {err}") from err
    db = Database(con, path)
    return db.table(table) if table is not None else db


file_format("sqlite", (".sqlite", ".sqlite3"), _read_sqlite, None,
            needs_table=True)
