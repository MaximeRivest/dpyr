"""Errors with dplyr-grade messages (ROADMAP 1.4).

The schema-eager promise (DESIGN.md §3): every user mistake surfaces on the
line that made it, with a short message and a did-you-mean suggestion.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable


class DpyrError(Exception):
    """Base class for all dpyr errors."""


class ColumnNotFoundError(DpyrError, KeyError):
    def __init__(self, name: str, available: Iterable[str], context: str = "") -> None:
        self.name = name
        self.available = list(available)
        where = f" in {context}" if context else ""
        msg = f"column '{name}' not found{where}."
        close = difflib.get_close_matches(name, self.available, n=1)
        if close:
            msg += f" Did you mean '{close[0]}'?"
        msg += f" Available columns: {', '.join(self.available) or '(none)'}"
        super().__init__(msg)

    def __str__(self) -> str:  # KeyError quotes its arg; we want the message
        return self.args[0]


class ExprTypeError(DpyrError, TypeError):
    """An expression was applied to a column of the wrong dtype."""


class DuplicateColumnError(DpyrError, ValueError):
    def __init__(self, name: str, context: str) -> None:
        super().__init__(f"duplicate column '{name}' in {context}")


class GroupError(DpyrError, ValueError):
    """Invalid operation for the frame's grouping state."""
