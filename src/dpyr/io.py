"""readr-flavored persistence (1.2.0): files in, frames out, no connection
objects in your face.

- read_duckdb("warehouse.db") opens the whole catalog as a Database object
  whose attributes are frames.
- write_duckdb / to_table / to_view land results in the engine without the
  data ever entering Python.
- read_ipc memory-maps Arrow IPC files (zero-copy from disk).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import DpyrError

if TYPE_CHECKING:
    import duckdb

    from .frame import DFrame

_FILE_CONS: dict[str, Any] = {}


def _file_con(path: str) -> duckdb.DuckDBPyConnection:
    """One shared connection per database file (duckdb is single-writer)."""
    import os

    import duckdb
    key = os.path.abspath(path)
    if key not in _FILE_CONS:
        _FILE_CONS[key] = duckdb.connect(key)
    return _FILE_CONS[key]


class Database:
    """A duckdb file opened the readr way: tables are attributes.

        db = read_duckdb("warehouse.db")
        db.tables            # ['customers', 'orders']
        db.orders            # a DFrame, schema known, lazy
        db.sql("SELECT 42")  # escape hatch
    """

    con: duckdb.DuckDBPyConnection

    def __init__(self, con: duckdb.DuckDBPyConnection, label: str) -> None:
        object.__setattr__(self, "con", con)
        object.__setattr__(self, "_label", label)

    @property
    def tables(self) -> list[str]:
        rows = self.con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema IN ('main', 'temp') ORDER BY table_name"
        ).fetchall()
        return [r[0] for r in rows if not r[0].startswith("__dpyr_")]

    def table(self, name: str) -> DFrame:
        from .frame import from_duckdb
        return from_duckdb(self.con, name)

    def sql(self, query: str, name: str = "sql") -> DFrame:
        from .frame import read_sql
        return read_sql(self.con, query, name=name)

    def __getattr__(self, name: str) -> DFrame:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self.tables:
            from .errors import ColumnNotFoundError
            raise ColumnNotFoundError(name, self.tables,
                                      f"database {self._label!r}")
        return self.table(name)

    def __getitem__(self, name: str) -> DFrame:
        return self.__getattr__(name)

    def __dir__(self) -> list[str]:
        return [*super().__dir__(), *self.tables]

    def __repr__(self) -> str:
        lines = [f"# duckdb database: {self._label}"]
        for t in self.tables:
            cols = self.con.execute(
                f'DESCRIBE "{t}"').fetchall()
            shown = ", ".join(f"{c[0]} <{c[1].lower()}>" for c in cols[:6])
            more = ", …" if len(cols) > 6 else ""
            lines.append(f"#   {t}: {shown}{more}")
        return "\n".join(lines)


def read_duckdb(path: str, table: str | None = None) -> Database | DFrame:
    """Open a duckdb file. With a table name, returns that frame directly;
    without, returns the catalog as a Database object."""
    import os
    if not os.path.exists(path):
        raise DpyrError(f"read_duckdb: no such file {path!r} "
                        "(use write_duckdb/to_table to create one)")
    con = _file_con(path)
    db = Database(con, path)
    return db.table(table) if table is not None else db


_DB_SUFFIXES = (".db", ".duckdb", ".ddb")
_IPC_SUFFIXES = (".arrow", ".feather", ".ipc")


def read(source: Any, table: str | None = None) -> DFrame | Database:
    """The one way in. Paths dispatch on extension; everything else
    dispatches on type:

        read("orders.parquet")            # parquet scan (globs work)
        read("orders.csv")                # csv scan
        read("orders.arrow")              # arrow IPC, memory-mapped
        read("shop.db")                   # duckdb file -> Database catalog
        read("shop.db", "orders")         # one duckdb table -> frame
        read({"x": [1, 2]})               # plain Python data
        read(polars_or_pandas_dataframe)  # zero/near-zero copy
        read(arrow_table)
        read(duckdb_connection)           # live connection -> Database
        read(duckdb_connection, "orders") # one table on that connection
    """
    import duckdb

    from .frame import from_dict, from_polars
    if isinstance(source, dict):
        if table is not None:
            raise DpyrError("read(table=...) only applies to duckdb sources")
        return from_dict(source)
    if isinstance(source, duckdb.DuckDBPyConnection):
        db = Database(source, "connection")
        return db.table(table) if table is not None else db
    if not isinstance(source, str):
        if table is not None:
            raise DpyrError("read(table=...) only applies to duckdb sources")
        import polars as pl
        if isinstance(source, (pl.DataFrame, pl.LazyFrame)):
            return from_polars(source)
        type_name = f"{type(source).__module__}.{type(source).__name__}"
        if type_name.startswith("pandas."):
            return from_polars(pl.from_pandas(source))
        if type_name.startswith("pyarrow."):
            out = pl.from_arrow(source)
            assert isinstance(out, pl.DataFrame)
            return from_polars(out)
        raise DpyrError(
            f"read() doesn't know what to do with {type_name}; give it a "
            "path, dict, polars/pandas frame, arrow table, or duckdb "
            "connection")
    import pathlib
    suffix = pathlib.PurePath(source).suffix.lower()
    if suffix in _DB_SUFFIXES:
        return read_duckdb(source, table)
    if table is not None:
        raise DpyrError(
            f"read(table=...) only applies to duckdb sources, not {suffix!r}")
    from .frame import read_csv, read_parquet
    if suffix in (".parquet", ".pq"):
        return read_parquet(source)
    if suffix == ".csv":
        return read_csv(source)
    if suffix in _IPC_SUFFIXES:
        return read_ipc(source)
    raise DpyrError(
        f"read() can't infer a format from {source!r} (suffix {suffix!r}); "
        "supported: .parquet/.pq, .csv, .arrow/.feather/.ipc, "
        ".db/.duckdb/.ddb")


def read_ipc(path: str) -> DFrame:
    """Read an Arrow IPC (Feather v2) file — memory-mapped, zero-copy."""
    import polars as pl

    from . import plan as p
    from .backend import _REGISTRY, PolarsPayload
    from .frame import DFrame, _file_token
    from .polars_backend import _normalize, schema_from_polars
    lf = _normalize(pl.scan_ipc(path, memory_map=True))
    token = _file_token("ipc", path)
    _REGISTRY[token] = PolarsPayload(lf)
    return DFrame(p.Source(path, tuple(schema_from_polars(lf).items()), token))
