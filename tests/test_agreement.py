"""Backend agreement fuzzing: polars and duckdb must produce identical
results for every plan (TESTING.md property 2 — our most important
invariant).

Strategy: generate a random frame (mixed dtypes, nulls, duplicates, empty)
and a random well-typed verb chain; run it on both backends; compare
exactly. Where the spec leaves row order undefined, the chain ends with an
arrange over all columns, so order is always pinned before comparison.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

import dpyr as d
from dpyr import coalesce, col, cum_sum, dense_rank, if_else, lag, lead, min_rank, n, row_number

# -- data ----------------------------------------------------------------

ints = st.one_of(st.none(), st.integers(-20, 20))
floats = st.one_of(st.none(), st.floats(-50, 50, allow_nan=False).map(
    lambda x: round(x, 3)))
strs = st.one_of(st.none(), st.sampled_from(["a", "b", "Bb", "cc", "", "zz"]))
bools = st.one_of(st.none(), st.booleans())

COLS: dict[str, tuple[st.SearchStrategy, d.DType]] = {
    "i": (ints, d.INT64), "j": (ints, d.INT64),
    "x": (floats, d.FLOAT64), "y": (floats, d.FLOAT64),
    "s": (strs, d.STR), "t": (strs, d.STR),
    "b": (bools, d.BOOL),
}


@st.composite
def tables(draw, min_rows: int = 0) -> dict:
    n_rows = draw(st.integers(min_rows, 8))
    names = draw(st.lists(st.sampled_from(list(COLS)), min_size=2,
                          max_size=5, unique=True))
    out: dict = {}
    for name in names:
        gen, dtype = COLS[name]
        values = [draw(gen) for _ in range(n_rows)]
        # polars infers all-null columns as Null; give duckdb the same hint
        # by making fully-null columns rare but allowed
        out[name] = values
    return out


def _typed_polars(data: dict):
    import polars as pl
    to_pl = {d.INT64: pl.Int64, d.FLOAT64: pl.Float64, d.STR: pl.String,
             d.BOOL: pl.Boolean}
    schema = {k: to_pl[COLS[k][1]] for k in data if k in COLS}
    return pl.DataFrame(data, schema_overrides=schema)


def make_pair(data: dict) -> tuple[d.DFrame, d.DFrame]:
    df = _typed_polars(data)
    return d.from_polars(df), _duck_from_polars(df)


def _duck_from_polars(df) -> d.DFrame:
    from conftest import _con, _table_counter
    _table_counter[0] += 1
    name = f"agree{_table_counter[0]}"
    _con.register(f"{name}_arrow", df.to_arrow())
    _con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "{name}_arrow"')
    _con.unregister(f"{name}_arrow")
    return d.from_duckdb(_con, name)


# -- chains ----------------------------------------------------------------

def _num_cols(f: d.DFrame) -> list[str]:
    return [k for k, v in f.schema.items() if d.dtypes.is_numeric(v)]


def _str_cols(f: d.DFrame) -> list[str]:
    return [k for k, v in f.schema.items() if v == d.STR]


@st.composite
def verb(draw, f: d.DFrame):
    """One random verb applicable to f, as a function DFrame -> DFrame."""
    nums, strings = _num_cols(f), _str_cols(f)
    options = ["filter_num", "filter_isna", "mutate_arith", "mutate_ifelse",
               "select", "distinct", "head", "tail", "summarize", "grouped_mutate",
               "window", "grouped_window", "slice_minmax"]
    if strings:
        options += ["filter_str", "mutate_str"]
    choice = draw(st.sampled_from(options))

    if choice == "filter_num" and nums:
        c, k = draw(st.sampled_from(nums)), draw(st.integers(-10, 10))
        return lambda fr: fr.filter(col[c] > k)
    if choice == "filter_isna":
        c = draw(st.sampled_from(list(f.schema)))
        neg = draw(st.booleans())
        return lambda fr: fr.filter(~col[c].is_na() if neg else col[c].is_na())
    if choice == "filter_str":
        c = draw(st.sampled_from(strings))
        pat = draw(st.sampled_from(["a", "b", "^c"]))
        return lambda fr: fr.filter(col[c].str_detect(pat))
    if choice == "mutate_arith" and nums:
        c = draw(st.sampled_from(nums))
        k = draw(st.integers(1, 5))
        op = draw(st.sampled_from(["add", "div", "floordiv", "mod"]))
        e = {"add": col[c] + k, "div": col[c] / k,
             "floordiv": col[c] // k, "mod": col[c] % k}[op]
        name = f"m{draw(st.integers(0, 99))}"
        return lambda fr, e=e, name=name: fr.mutate(**{name: e})
    if choice == "mutate_ifelse" and nums:
        c = draw(st.sampled_from(nums))
        return lambda fr: fr.mutate(flag=if_else(col[c] > 0, 1, 0))
    if choice == "mutate_str" and strings:
        c = draw(st.sampled_from(strings))
        return lambda fr: fr.mutate(up=col[c].str_to_upper(),
                                    ln=col[c].str_len())
    if choice == "select":
        keep = draw(st.lists(st.sampled_from(list(f.schema)), min_size=1,
                             max_size=len(f.schema), unique=True))
        return lambda fr: fr.select(*keep)
    if choice == "distinct":
        return lambda fr: fr.distinct()
    if choice in ("head", "tail"):
        k = draw(st.integers(0, 5))
        keys = [col[c] for c in f.schema]
        if choice == "head":
            return lambda fr: fr.arrange(*keys).slice_head(k)
        return lambda fr: fr.arrange(*keys).slice_tail(k)
    if choice == "summarize" and nums:
        key = draw(st.sampled_from(list(f.schema)))
        c = draw(st.sampled_from(nums))
        agg = draw(st.sampled_from(["mean", "sum", "min", "max", "n_unique"]))
        e = getattr(col[c], agg)()

        def fresh(base: str) -> str:  # deterministic from schema: same on both
            name = base
            while name in f.schema or name == key:
                name += "_"
            return name

        names = {fresh("cnt"): n(), fresh(f"a_{agg}"): e}
        return lambda fr, names=names, key=key: (
            fr.group_by(col[key]).summarize(**names))
    if choice == "window" and nums:
        c = draw(st.sampled_from(nums))
        kind = draw(st.sampled_from(["lag", "lead", "rank", "dense", "cum", "rn", "coal"]))
        e = {"lag": lag(col[c]), "lead": lead(col[c], default=0),
             "rank": min_rank(col[c]), "dense": dense_rank(col[c]),
             "cum": cum_sum(col[c]), "rn": row_number(),
             "coal": coalesce(col[c], -1)}[kind]
        name = f"w{draw(st.integers(0, 99))}"
        # pin row order first so the window result is deterministic
        keys = [col[k] for k in f.schema]
        return lambda fr, e=e, name=name: fr.arrange(*keys).mutate(**{name: e})
    if choice == "grouped_window" and nums:
        key = draw(st.sampled_from(list(f.schema)))
        c = draw(st.sampled_from(nums))
        e = draw(st.sampled_from([lag(col[c]), cum_sum(col[c]), min_rank(col[c])]))
        name = f"gw{draw(st.integers(0, 99))}"
        keys = [col[k] for k in f.schema]
        return lambda fr, e=e, name=name, key=key: (
            fr.arrange(*keys).group_by(col[key]).mutate(**{name: e}).ungroup())
    if choice == "slice_minmax" and nums:
        c = draw(st.sampled_from(nums))
        k = draw(st.integers(1, 4))
        if draw(st.booleans()):
            return lambda fr, c=c, k=k: fr.slice_min(col[c], k)
        return lambda fr, c=c, k=k: fr.slice_max(col[c], k)
    if choice == "grouped_mutate" and nums:
        key = draw(st.sampled_from(list(f.schema)))
        c = draw(st.sampled_from(nums))
        return lambda fr: (fr.group_by(col[key])
                           .mutate(gm=col[c] - col[c].mean()).ungroup())
    return lambda fr: fr  # fallthrough no-op


def _final_sort(f: d.DFrame) -> d.DFrame:
    if isinstance(f, d.GroupedDFrame):
        f = f.ungroup()
    return f.arrange(*[col[c] for c in f.schema])


@settings(max_examples=120, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(st.data())
def test_backends_agree_on_random_chains(data):
    table = data.draw(tables())
    pf, df = make_pair(table)
    steps = data.draw(st.integers(1, 4))
    for _ in range(steps):
        step = data.draw(verb(pf))
        pf2, df2 = step(pf), step(df)
        # keep chains alive: skip steps that empty the schema entirely
        pf, df = pf2, df2
    a = _final_sort(pf).collect()
    b = _final_sort(df).collect()
    assert_frame_equal(a, b, rel_tol=1e-9, abs_tol=1e-12)


JOIN_LEFT = {"k": [1, 2, 2, None, 5], "v": [1.0, None, 3.0, 4.0, 5.0]}
JOIN_RIGHT = {"k": [2, 2, None, 6], "w": ["a", "b", "c", None]}


@settings(max_examples=30, deadline=None)
@given(how=st.sampled_from(["inner", "left", "right", "full", "semi", "anti"]),
       na=st.sampled_from(["na", "never"]))
def test_backends_agree_on_joins(how, na):
    pa, da = make_pair(JOIN_LEFT)
    pb, db = make_pair(JOIN_RIGHT)
    if how in ("semi", "anti"):
        pj = getattr(pa, f"{how}_join")(pb, on=col.k)
        dj = getattr(da, f"{how}_join")(db, on=col.k)
    else:
        pj = getattr(pa, f"{how}_join")(pb, on=col.k, na_matches=na)
        dj = getattr(da, f"{how}_join")(db, on=col.k, na_matches=na)
    assert_frame_equal(_final_sort(pj).collect(), _final_sort(dj).collect(),
                       rel_tol=1e-9)


@settings(max_examples=40, deadline=None)
@given(data=tables(min_rows=1))
def test_backends_agree_on_pivot_longer(data):
    pf, df = make_pair(data)
    nums = _num_cols(pf)
    if len(nums) < 2:
        return
    p_out = _final_sort(pf.mutate(**{c: col[c] / 1 for c in nums})
                        .pivot_longer([col[c] for c in nums])).collect()
    d_out = _final_sort(df.mutate(**{c: col[c] / 1 for c in nums})
                        .pivot_longer([col[c] for c in nums])).collect()
    assert_frame_equal(p_out, d_out, rel_tol=1e-9)
