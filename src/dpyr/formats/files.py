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

def _xlsx_sheet_names(path: str, mode: str) -> list[str]:
    import os
    try:
        import fastexcel
    except (ImportError, ModuleNotFoundError) as err:
        raise DpyrError(
            f"{mode} .xlsx needs the excel extra: pip install 'dpyr[excel]'"
        ) from err
    if not os.path.exists(path):
        raise DpyrError(f"read({path!r}): no such file")
    return list(fastexcel.read_excel(path).sheet_names)


def _read_xlsx_sheet(path: str, sheet: str) -> DFrame:
    import polars as pl
    df = pl.read_excel(path, sheet_name=sheet)
    # the sheet is part of the token: two sheets of one file must never
    # share a plan hash (the cache would serve one sheet's rows as the other)
    return _scan_source(f"xlsx[{sheet}]", df.lazy(), path)


class Workbook:
    """A multi-sheet .xlsx file opened the catalog way, like a duckdb
    Database: sheets are attributes, with completion and did-you-mean.

        wb = read("report.xlsx")
        wb.sheets        # ['2024 plots', 'notes']
        wb["2024 plots"] # a frame; wb.notes works for plain names
    """

    _path: str
    _names: list[str]
    _label: str

    def __init__(self, path: str, names: list[str],
                 label: str | None = None) -> None:
        self._path = path
        self._names = names
        # what messages show: the original URL for a downloaded Google
        # Sheet, the path itself otherwise
        self._label = label or path

    @property
    def sheets(self) -> list[str]:
        return list(self._names)

    def sheet(self, name: str) -> DFrame:
        if name not in self._names:
            import difflib
            close = difflib.get_close_matches(name, self._names, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise DpyrError(
                f"no sheet named {name!r} in {self._label!r}.{hint} "
                f"Sheets: {', '.join(self._names)}")
        return _read_xlsx_sheet(self._path, name)

    def __getattr__(self, name: str) -> DFrame:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.sheet(name)

    def __getitem__(self, name: str) -> DFrame:
        return self.sheet(name)

    def __dir__(self) -> list[str]:
        return [*super().__dir__(), *self._names]

    def __repr__(self) -> str:
        lines = [f"# excel workbook: {self._label}"]
        lines += [f"#   {s}" for s in self._names]
        lines.append("# read a sheet: wb.sheet(name), wb[name], or wb.<name>")
        return "\n".join(lines)


def _read_xlsx(path: str, table: str | None,
               label: str | None = None) -> DFrame | Workbook:
    names = _xlsx_sheet_names(path, "reading")
    if table is None:
        # one sheet behaves like a CSV; several open as a catalog
        if len(names) == 1:
            return _read_xlsx_sheet(path, names[0])
        return Workbook(path, names, label)
    return Workbook(path, names, label).sheet(table)


# -- google sheets (read-only, via the workbook's xlsx export) ---------------------

def is_gsheet_url(source: str) -> bool:
    return "docs.google.com/spreadsheets" in source


def read_gsheet(url: str, table: str | None) -> DFrame | Workbook:
    """Read a Google Sheet by its ordinary browser URL. The whole workbook
    is fetched through Google's xlsx export, so sheets behave exactly like
    a local Excel file (single sheet -> frame, several -> Workbook).
    Works for link-readable sheets; private ones explain how to share."""
    import re
    import tempfile
    import urllib.error
    import urllib.request

    m = re.search(r"docs\.google\.com/spreadsheets/d/([\w-]+)", url)
    if m is None:
        raise DpyrError(
            f"read() can't find a spreadsheet id in {url!r}; expected a URL "
            "like https://docs.google.com/spreadsheets/d/<id>/...")
    export = (f"https://docs.google.com/spreadsheets/d/{m.group(1)}"
              "/export?format=xlsx")
    not_shared = DpyrError(
        f"this Google Sheet is not link-readable: {url!r}. In Google "
        "Sheets use Share -> 'Anyone with the link' (Viewer), or "
        "File -> Download -> .xlsx and read the file")
    req = urllib.request.Request(export, headers={"User-Agent": "dpyr"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
    except urllib.error.HTTPError as err:
        raise not_shared from err
    if not data.startswith(b"PK"):  # the login page, not an xlsx (zip)
        raise not_shared
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as fh:
        fh.write(data)
    return _read_xlsx(fh.name, table, label=url)


def _write_xlsx(frame: DFrame, path: str, table: str | None) -> None:
    import os
    try:
        import polars as pl
        import xlsxwriter  # type: ignore[import-untyped]
    except (ImportError, ModuleNotFoundError) as err:
        raise DpyrError(
            "writing .xlsx needs the excel extra: pip install 'dpyr[excel]'"
        ) from err
    target = table or "Sheet1"
    df = frame.collect()
    # writing into an existing workbook replaces only the target sheet;
    # the others are carried over (values, not cell formatting)
    existing = _xlsx_sheet_names(path, "writing") if os.path.exists(path) else []
    others = [n for n in existing if n != target]
    if others:
        import warnings
        warnings.warn(
            f"write({path!r}, {target!r}): keeping the workbook's other "
            f"sheets ({', '.join(others)}) — values are preserved, cell "
            "formatting is not", stacklevel=3)
    ordered = [(n, df if n == target else pl.read_excel(path, sheet_name=n))
               for n in existing]  # keep the original tab order
    if target not in existing:
        ordered.append((target, df))
    with xlsxwriter.Workbook(path) as wb:
        for name, data in ordered:
            data.write_excel(workbook=wb, worksheet=name)


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
