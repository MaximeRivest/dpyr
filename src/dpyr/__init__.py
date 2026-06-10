"""dataframe — dplyr for Python, fronting polars and duckdb.

This release reserves the package name while the library is under active
development. See https://github.com/maximerivest/dpyr for the design
documents and roadmap.
"""

__version__ = "0.0.1"


def __getattr__(name: str):
    raise NotImplementedError(
        f"dpyr.{name} is not available yet: version {__version__} is a "
        "name-reservation release. The dplyr-style API (filter, mutate, "
        "group_by, summarize, ...) is under development — see "
        "https://github.com/maximerivest/dpyr for the roadmap."
    )
