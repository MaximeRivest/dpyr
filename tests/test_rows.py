"""Row functions: any Python function, once per row, as a column (S38)."""

from __future__ import annotations

import dataclasses
import enum
import threading
import time
from typing import Literal, Optional

import polars as pl
import pytest

import dpyr as d
from dpyr import col, lag, n, vectorize

WORDS = {"t": ["b", "a", "c", "a"], "g": ["x", "y", "x", "y"], "k": [1, 2, 3, 4]}


def counted(fn):
    """A vectorized function that records every call it gets."""
    calls: list = []

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return fn(*args, **kwargs)

    wrapper.__annotations__ = fn.__annotations__
    wrapper.__name__ = fn.__name__
    rf = vectorize(wrapper)
    rf.calls = calls  # type: ignore[attr-defined]
    return rf


def test_one_call_per_distinct_row_and_the_type_from_the_annotation(make):
    def shout(text: str) -> str:
        return text.upper()
    f = counted(shout)
    out = make(WORDS).mutate(u=f(col.t))
    assert out.schema["u"] == d.STR
    assert out.collect()["u"].to_list() == ["B", "A", "C", "A"]
    assert len(f.calls) == 3                                  # "a" twice, computed once


def test_several_inputs_by_position_or_name_and_constants(make):
    @vectorize
    def join(a: str, b: str, sep: str = "-") -> str:
        return a + sep + b

    got = make(WORDS).mutate(j=join(col.t, col.g), k2=join(col.t, b=col.g, sep="|")).collect()
    assert got["j"].to_list() == ["b-x", "a-y", "c-x", "a-y"]
    assert got["k2"].to_list() == ["b|x", "a|y", "c|x", "a|y"]


def test_arguments_can_be_expressions_and_results_feed_expressions(make):
    @vectorize
    def double(x: int) -> int:
        return 2 * x

    got = make(WORDS).mutate(a=double(col.k + 1), b=double(double(col.k)) > 10,
                             c=col.a * 10).collect()
    assert got["a"].to_list() == [4, 6, 8, 10]
    assert got["b"].to_list() == [False, False, True, True]
    assert got["c"].to_list() == [40, 60, 80, 100]
    assert list(got.columns) == ["t", "g", "k", "a", "b", "c"]           # no helper columns


def test_filter_with_a_row_function(make):
    @vectorize
    def vowel(t: str) -> bool:
        return t in "aeiou"

    assert make(WORDS).filter(vowel(col.t)).collect()["k"].to_list() == [2, 4]


def test_grouped_mutate_keeps_the_groups(make):
    @vectorize
    def shout(t: str) -> str:
        return t.upper()

    f = make(WORDS).group_by(col.g).mutate(u=shout(col.t), m=n(), prev=lag(col.k))
    assert f.groups == ("g",)
    got = f.ungroup().arrange(col.k).collect()
    assert got["u"].to_list() == ["B", "A", "C", "A"]
    assert got["prev"].to_list() == [None, None, 1, 2]
    assert got["m"].to_list() == [2, 2, 2, 2]


def test_displaying_runs_only_the_shown_rows_and_results_are_remembered(make):
    def shout(t: str) -> str:
        return t.upper()
    f = counted(shout)
    big = make({"t": [f"w{i}" for i in range(40)]})
    frame = big.mutate(u=f(col.t)).select(col.u)
    text = repr(frame)
    assert "showing 10 of 40 rows" in text and len(f.calls) == 10
    frame.collect()
    assert len(f.calls) == 40                     # the 10 shown are not asked again
    big.mutate(u=f(col.t)).filter(col.u != "W0").collect()
    assert len(f.calls) == 40                     # a new chain, the same answers


def test_threads_run_rows_concurrently(make):
    active, peak = [0], [0]
    lock = threading.Lock()

    @vectorize(threads=8)
    def slow(t: str) -> str:
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        return t

    make({"t": [f"w{i}" for i in range(16)]}).mutate(u=slow(col.t)).collect()
    assert peak[0] > 1


def test_failures_raise_after_every_row_ran_and_a_rerun_retries_only_them(make):
    attempts: list = []
    broken = {"c"}

    @vectorize
    def flaky(t: str) -> str:
        attempts.append(t)
        if t in broken:
            raise ValueError("down")
        return t.upper()

    frame = make(WORDS).mutate(u=flaky(col.t))
    with pytest.raises(d.DpyrError, match=r"failed on 1 of 4 rows \(row 2: ValueError: down\)"):
        frame.collect()
    assert sorted(attempts) == ["a", "b", "c"]
    broken.clear()
    assert frame.collect()["u"].to_list() == ["B", "A", "C", "A"]
    assert sorted(attempts) == ["a", "b", "c", "c"]


def test_errors_null_keeps_going_with_a_warning(make):
    @vectorize(errors="null")
    def flaky(t: str) -> str:
        if t == "c":
            raise ValueError("down")
        return t

    with pytest.warns(UserWarning, match="failed on 1 of 4 rows"):
        got = make(WORDS).mutate(u=flaky(col.t)).collect()
    assert got["u"].to_list() == ["b", "a", None, "a"]


@dataclasses.dataclass
class Entity:
    name: str
    score: float


class Tone(enum.Enum):
    HAPPY = "happy"
    SAD = "sad"


def test_structured_results_become_nested_or_scalar_columns(make):
    @vectorize
    def entities(t: str) -> list[Entity]:
        return [Entity(t, 1.0)]

    @vectorize
    def tone(t: str) -> Tone:
        return Tone.HAPPY

    @vectorize
    def maybe(t: str) -> Optional[Literal["yes", "no"]]:
        return None if t == "a" else "yes"

    f = make(WORDS).mutate(e=entities(col.t), tone=tone(col.t), m=maybe(col.t))
    assert f.schema["e"] == d.dtypes.nested("List(Struct(name: Str, score: Float64))")
    assert f.schema["tone"] == d.STR and f.schema["m"] == d.STR
    got = f.collect()
    assert got["e"].to_list()[0] == [{"name": "b", "score": 1.0}]
    assert got["tone"].to_list()[0] == "happy" and got["m"].to_list() == ["yes", None, "yes", None]


def test_types_are_checked_when_the_chain_is_written(make):
    with pytest.raises(d.ExprTypeError, match="annotate it"):
        vectorize(lambda t: t)
    f = vectorize(lambda t: t, dtype=str)
    frame = make(WORDS)
    with pytest.raises(d.ColumnNotFoundError):
        frame.mutate(u=f(col.nope))
    with pytest.raises(d.ExprTypeError, match="S38"):
        frame.summarize(u=f(col.t))
    with pytest.raises(d.ExprTypeError, match="S38"):
        frame.arrange(f(col.t))
    with pytest.raises(d.ExprTypeError, match=r"cannot apply \+ to Str and Int64 in \(<lambda>\(col.t\)"):
        frame.mutate(u=f(col.t) + 1)                       # the message shows what you wrote
    with pytest.raises(d.ColumnNotFoundError, match="Available columns: t, g, k, a$"):
        frame.mutate(a=f(col.t), b=col.nope)


def test_a_result_that_does_not_fit_names_the_row(make):
    @vectorize
    def wrong(t: str) -> int:
        return "oops" if t == "c" else 1  # type: ignore[return-value]

    with pytest.raises(d.DpyrError, match="returned str 'oops' on row 2"):
        make(WORDS).mutate(u=wrong(col.t)).collect()


def test_plain_values_call_the_function_directly():
    @vectorize
    def shout(t: str) -> str:
        return t.upper()

    assert shout("hi") == "HI"


def test_version_separates_remembered_results(make):
    def answer(t: str) -> str:
        return state["reply"]
    state = {"reply": "one"}
    first = vectorize(answer, version="1")
    assert make(WORDS).mutate(u=first(col.t)).collect()["u"][0] == "one"
    state["reply"] = "two"
    assert make(WORDS).mutate(u=vectorize(answer, version="1")(col.t)).collect()["u"][0] == "one"
    assert make(WORDS).mutate(u=vectorize(answer, version="2")(col.t)).collect()["u"][0] == "two"


def test_persist_write_and_to_table_run_the_rows(make, tmp_path):
    @vectorize
    def shout(t: str) -> str:
        return t.upper()

    frame = make(WORDS).mutate(u=shout(col.t))
    assert frame.persist().collect()["u"].to_list() == ["B", "A", "C", "A"]
    frame.write(str(tmp_path / "x.parquet"))
    assert pl.read_parquet(tmp_path / "x.parquet")["u"].to_list() == ["B", "A", "C", "A"]
    with pytest.raises(d.DpyrError, match="per row"):
        frame.show_query()
