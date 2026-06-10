"""dpyr — dplyr for Python, fronting polars and duckdb.

dplyr's verbs, Python's method chains, real autocompletion, and two
interchangeable execution backends. See https://github.com/MaximeRivest/dpyr
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
from .expr import (
    BoolExpr,
    Expr,
    NumExpr,
    StrExpr,
    TemporalExpr,
    case_when,
    col,
    desc,
    if_else,
    infer_dtype,
    lit,
    n,
    typed_col,
)
from .frame import (
    DFrame,
    GroupedDFrame,
    from_dict,
    from_duckdb,
    from_pandas,
    from_polars,
    from_schema,
    read_csv,
    read_parquet,
    read_sql,
)
from .materialize import cache_clear, cache_size, options
from .plan import plan_hash
from .tidyselect import (
    across,
    contains,
    ends_with,
    everything,
    is_bool,
    is_numeric,
    is_string,
    matches,
    starts_with,
    where,
)

__version__ = "1.0.0"

__all__ = [
    # frame + sources
    "DFrame", "GroupedDFrame", "from_schema", "from_polars", "from_dict",
    "from_pandas", "read_parquet", "read_csv", "from_duckdb", "read_sql",
    # expressions
    "col", "n", "desc", "if_else", "case_when", "lit", "Expr", "typed_col",
    "NumExpr", "StrExpr", "BoolExpr", "TemporalExpr", "infer_dtype",
    # tidyselect / across
    "across", "starts_with", "ends_with", "contains", "matches", "where",
    "everything", "is_numeric", "is_string", "is_bool",
    # dtypes
    "DType", "INT64", "FLOAT64", "BOOL", "STR", "DATE", "DATETIME", "NULL",
    "dtypes",
    # materialization
    "options", "cache_clear", "cache_size", "plan_hash",
    # errors
    "DpyrError", "ColumnNotFoundError", "ExprTypeError",
    "DuplicateColumnError", "GroupError",
]
