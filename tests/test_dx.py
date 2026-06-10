"""Developer-experience tests (Epic 7): df.c proxy, lambda verbs, stubgen,
short tracebacks."""

from __future__ import annotations

import traceback
from pathlib import Path

import polars as pl
import pytest

import dpyr as d
from dpyr import col
from dpyr.cli import generate, main
from dpyr.expr import NumExpr, StrExpr

from conftest import STARWARS


def test_frame_bound_proxy_is_typed(make):
    f = make(STARWARS)
    assert isinstance(f.c.height, NumExpr)
    assert isinstance(f.c.name, StrExpr)
    assert "height" in dir(f.c)


def test_proxy_unknown_column_suggests(make):
    with pytest.raises(d.ColumnNotFoundError, match="Did you mean 'height'"):
        make(STARWARS).c.heigth


def test_proxy_blocks_wrong_typed_method_at_build_time(make):
    with pytest.raises(d.ExprTypeError, match="not available on a StrExpr"):
        make(STARWARS).c.name.mean()


def test_lambda_predicates_and_mutate(make):
    f = make(STARWARS)
    out = (f.filter(lambda c: c.height > 180)
            .mutate(double=lambda c: c.height * 2)
            .collect())
    assert out["name"].to_list() == ["Chewie"]
    assert out["double"].to_list() == [456]


def test_traceback_is_short(make):
    f = make(STARWARS)
    try:
        f.filter(col.heigth > 100)
    except d.ColumnNotFoundError as e:
        tb = traceback.extract_tb(e.__traceback__)
        assert len(tb) <= 2, f"traceback too deep: {[fr.name for fr in tb]}"
        assert e.__cause__ is None and e.__context__ is None
    else:
        raise AssertionError("expected ColumnNotFoundError")


def test_stubgen_generates_typed_module(tmp_path: Path):
    data = pl.DataFrame({"height": [1], "name": ["x"], "alive": [True]})
    pq = tmp_path / "starwars.parquet"
    data.write_parquet(pq)
    code = generate([pq])
    assert "class StarwarsCols(ColsProxy):" in code
    assert "height: NumExpr" in code
    assert "name: StrExpr" in code
    assert "alive: BoolExpr" in code
    assert "starwars: DFrame = load_starwars()" in code
    # generated module actually imports and runs
    ns: dict = {}
    exec(compile(code, "gen", "exec"), ns)  # noqa: S102
    assert ns["starwars"].columns == ["height", "name", "alive"]


def test_stubgen_cli_writes_file(tmp_path: Path, capsys):
    data = pl.DataFrame({"x": [1.5]})
    pq = tmp_path / "t.parquet"
    data.write_parquet(pq)
    out = tmp_path / "schemas.py"
    assert main(["stubgen", str(pq), "-o", str(out)]) == 0
    assert "class TCols(ColsProxy):" in out.read_text()


def test_stubgen_sanitizes_bad_identifiers(tmp_path: Path):
    data = pl.DataFrame({"weird name": [1], "class": ["x"]})
    pq = tmp_path / "2bad.parquet"
    data.write_parquet(pq)
    code = generate([pq])
    assert "weird_name: NumExpr" in code
    assert "c_class: StrExpr" in code
