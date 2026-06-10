"""Differential tests against dplyr (Epic 4 / TESTING.md layer 1).

Each YAML spec under tests/specs/ carries a verb chain in two notations:
`r` (run by oracle/run_specs.R through real dplyr, producing the committed
golden parquet) and `py` (run here through dpyr on BOTH backends). The
comparison applies the normalizations documented in SEMANTICS.md:
- row order is not compared unless the spec sets check_order (S6 note);
  both sides are sorted by all columns first
- R parquet Int32 widens to Int64 (S5 canonical ints)
- float comparison uses a relative tolerance (S19)
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import polars as pl
import pytest
import yaml
from polars.testing import assert_frame_equal

import dpyr as d
from dpyr import (  # noqa: F401  (names available to spec py expressions)
    across,
    case_when,
    col,
    contains,
    desc,
    ends_with,
    everything,
    if_else,
    is_numeric,
    is_string,
    lit,
    matches,
    n,
    starts_with,
    where,
)

from conftest import make_duckdb, make_polars

HERE = Path(__file__).parent
SPEC_DIR = HERE / "specs"
GOLDEN_DIR = Path(os.environ.get("DPYR_GOLDEN_DIR", HERE / "golden"))

PY_TYPES = {"int": pl.Int64, "float": pl.Float64, "str": pl.String,
            "bool": pl.Boolean, "date": pl.Date}

_EVAL_NS = {name: globals()[name] for name in (
    "col", "n", "desc", "if_else", "case_when", "lit", "across",
    "starts_with", "ends_with", "contains", "matches", "where",
    "everything", "is_numeric", "is_string")}
for _name in ("lag", "lead", "row_number", "min_rank", "dense_rank",
              "percent_rank", "cum_sum", "cum_min", "cum_max", "coalesce",
              "replace_na"):
    _EVAL_NS[_name] = getattr(d, _name)
_EVAL_NS["FLOAT64"] = d.FLOAT64
_EVAL_NS["INT64"] = d.INT64
_EVAL_NS["STR"] = d.STR


def specs() -> list[Path]:
    return sorted(SPEC_DIR.rglob("*.yaml"))


def _build_data(data: dict, types: dict) -> pl.DataFrame:
    cols = {}
    for k, values in data.items():
        t = types[k]
        if t == "date":
            values = [datetime.date.fromisoformat(v) if isinstance(v, str) else v
                      for v in values]
        cols[k] = pl.Series(k, values, dtype=PY_TYPES[t])
    return pl.DataFrame(cols)


def _apply_step(frame: d.DFrame, step: dict, other: d.DFrame | None) -> d.DFrame:
    verb = step["verb"]
    if verb in ("filter", "mutate", "summarize", "arrange", "select", "rename"):
        code = f"frame.{verb}({step['py']})"
        return eval(code, {"frame": frame, **_EVAL_NS})  # noqa: S307
    if verb == "group_by":
        return frame.group_by(*step["cols"])
    if verb == "ungroup":
        return frame.ungroup()
    if verb == "distinct":
        return frame.distinct(*step.get("cols", []))
    if verb == "slice_head":
        return frame.slice_head(step["rows"])
    if verb == "slice_min":
        return frame.slice_min(col[step["order"]], step["rows"])
    if verb == "slice_max":
        return frame.slice_max(col[step["order"]], step["rows"])
    if verb == "separate":
        return frame.separate(step["column"], into=list(step["into"]),
                              sep=step.get("sep", "_"))
    if verb == "unite":
        return frame.unite(step["new"], list(step["cols"]),
                           sep=step.get("sep", "_"))
    if verb == "slice_tail":
        return frame.slice_tail(step["rows"])
    if verb == "count":
        return frame.count(*step.get("cols", []))
    if verb.endswith("_join"):
        assert other is not None
        by = step["by"]
        keys = by if isinstance(by, list) else [by]
        return getattr(frame, verb)(other, on=keys)
    if verb == "pivot_longer":
        return frame.pivot_longer(step["cols"],
                                  names_to=step.get("names_to", "name"),
                                  values_to=step.get("values_to", "value"))
    if verb == "pivot_wider":
        return frame.pivot_wider(names_from=step["names_from"],
                                 values_from=step["values_from"])
    raise AssertionError(f"unknown verb in spec: {verb}")


def _normalize(df: pl.DataFrame, check_order: bool) -> pl.DataFrame:
    casts = []
    for name, dtype in df.schema.items():
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16,
                     pl.UInt32, pl.UInt64):
            casts.append(pl.col(name).cast(pl.Int64))
        elif dtype == pl.Float32:
            casts.append(pl.col(name).cast(pl.Float64))
        elif dtype == pl.Float64:
            # S20: R yields NaN for mean of zero values where the engines
            # yield null; compare them as equal
            casts.append(pl.when(pl.col(name).is_nan()).then(None)
                         .otherwise(pl.col(name)).alias(name))
        elif isinstance(dtype, pl.Datetime):
            casts.append(pl.col(name).cast(pl.Datetime("us")))
        elif dtype == pl.Categorical:  # R factors (S17)
            casts.append(pl.col(name).cast(pl.String))
    if casts:
        df = df.with_columns(casts)
    if not check_order and df.width:
        df = df.sort(by=df.columns, nulls_last=True)
    return df


@pytest.mark.parametrize("spec_path", specs(), ids=lambda p: str(p.relative_to(SPEC_DIR)))
def test_matches_dplyr_golden(spec_path: Path, backend: str) -> None:
    spec = yaml.safe_load(spec_path.read_text())
    golden_path = GOLDEN_DIR / spec_path.relative_to(SPEC_DIR).with_suffix(".parquet")
    if not golden_path.exists():
        pytest.fail(f"golden missing: run `Rscript oracle/run_specs.R` "
                    f"(expected {golden_path})")

    make = make_polars if backend == "polars" else make_duckdb
    f = make(_build_data(spec["data"], spec["types"]))
    other = None
    if "data2" in spec:
        other = make(_build_data(spec["data2"], spec["types2"]))

    for step in spec["chain"]:
        f = _apply_step(f, step, other)
    if isinstance(f, d.GroupedDFrame):
        f = f.ungroup()

    got = _normalize(f.collect(), spec.get("check_order", False))
    want = _normalize(pl.read_parquet(golden_path), spec.get("check_order", False))
    assert_frame_equal(got, want, rel_tol=1e-9, abs_tol=1e-12,
                       check_column_order=True)


def test_meta_records_dplyr_version() -> None:
    meta = GOLDEN_DIR / "_meta.yaml"
    if not meta.exists():
        pytest.skip("goldens not generated yet")
    info = yaml.safe_load(meta.read_text())
    assert "dplyr" in info


if os.environ.get("DPYR_REQUIRE_GOLDENS") and not list(GOLDEN_DIR.rglob("*.parquet")):
    raise RuntimeError("DPYR_REQUIRE_GOLDENS set but tests/golden is empty")
