"""dpyr — dplyr for Python, fronting polars and duckdb.

Epic 1 (expression IR & schema engine) is implemented: build verb chains
against a schema and get immediate validation and output schemas. Execution
backends arrive in Epics 2 (polars) and 5 (duckdb).
"""

from . import dtypes
from .dtypes import BOOL, DATE, DATETIME, FLOAT64, INT64, NULL, STR, DType
from .errors import (
    ColumnNotFoundError,
    DpyrError,
    DuplicateColumnError,
    ExprTypeError,
    GroupError,
)
from .expr import Expr, case_when, col, desc, if_else, infer_dtype, lit, n
from .frame import DFrame, GroupedDFrame, from_schema
from .plan import plan_hash

__version__ = "0.1.0.dev0"

__all__ = [
    "col", "n", "desc", "if_else", "case_when", "lit",
    "DFrame", "GroupedDFrame", "from_schema",
    "DType", "INT64", "FLOAT64", "BOOL", "STR", "DATE", "DATETIME", "NULL",
    "dtypes", "Expr", "infer_dtype", "plan_hash",
    "DpyrError", "ColumnNotFoundError", "ExprTypeError",
    "DuplicateColumnError", "GroupError",
]
