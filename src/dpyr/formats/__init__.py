"""Format modules register themselves on import; the registry is the
single dispatch table for read()/write()."""

from . import files, objects  # noqa: F401  (registration side effects)
from .registry import match_file, match_object, readable, type_name, writable

__all__ = ["match_file", "match_object", "readable", "writable", "type_name"]
