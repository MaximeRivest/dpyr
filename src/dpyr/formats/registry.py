"""The format registry: every way in or out of dpyr is one registration.

Each format lives in its own module under dpyr.formats and registers
itself at import time. read()/write() are thin dispatchers over this
table — adding a format never touches them.

- file_format(): suffix-keyed readers/writers for paths and URLs
  (longest matching suffix wins, so ".csv.gz" beats ".gz").
- object_reader(): type-predicate-keyed readers for in-memory objects
  (polars/pandas frames, arrow tables, HF datasets, tensors, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..frame import DFrame
    from ..io import Database
    from .files import Workbook

ReadFn = Callable[[str, "str | None"], "DFrame | Database | Workbook"]
WriteFn = Callable[["DFrame", str, "str | None"], "DFrame | None"]


@dataclass(frozen=True)
class FileFormat:
    name: str
    suffixes: tuple[str, ...]
    reader: ReadFn | None
    writer: WriteFn | None
    needs_table: bool = False  # database-style destinations


_FILE_FORMATS: list[FileFormat] = []
_OBJECT_READERS: list[tuple[str, Callable[[Any], bool], Callable[[Any, Any], Any]]] = []


def file_format(name: str, suffixes: tuple[str, ...],
                reader: ReadFn | None = None, writer: WriteFn | None = None,
                needs_table: bool = False) -> None:
    _FILE_FORMATS.append(FileFormat(name, suffixes, reader, writer, needs_table))


def object_reader(name: str, predicate: Callable[[Any], bool],
                  reader: Callable[[Any, Any], Any]) -> None:
    _OBJECT_READERS.append((name, predicate, reader))


def match_file(path: str) -> FileFormat | None:
    lowered = path.lower().split("?")[0]  # ignore URL query strings
    best: FileFormat | None = None
    best_len = 0
    for fmt in _FILE_FORMATS:
        for suffix in fmt.suffixes:
            if lowered.endswith(suffix) and len(suffix) > best_len:
                best, best_len = fmt, len(suffix)
    return best


def match_object(source: Any) -> tuple[str, Callable[[Any, Any], Any]] | None:
    for name, predicate, reader in _OBJECT_READERS:
        if predicate(source):
            return name, reader
    return None


def readable() -> str:
    return ", ".join(sorted(
        "/".join(f.suffixes) for f in _FILE_FORMATS if f.reader))


def writable() -> str:
    return ", ".join(sorted(
        "/".join(f.suffixes) for f in _FILE_FORMATS if f.writer))


def type_name(source: Any) -> str:
    return f"{type(source).__module__}.{type(source).__name__}"
