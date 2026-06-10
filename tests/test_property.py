"""Hypothesis properties for the Epic-1 schema engine (TESTING.md layer 3.1).

Property 1 proper (predicted schema == collected schema) needs a backend;
until Epic 2 these check the engine's internal invariants: inference is
total over generated well-typed chains, deterministic, and hash-stable.
"""

from hypothesis import given, strategies as st

from dpyr import BOOL, FLOAT64, INT64, STR, col, desc, n, from_schema, plan_hash
from dpyr.dtypes import is_numeric

DTYPES = [INT64, FLOAT64, BOOL, STR]

names = st.sampled_from(["a", "b", "c", "d", "e", "f"])
schemas = st.dictionaries(names, st.sampled_from(DTYPES), min_size=1, max_size=6)


@st.composite
def frames(draw):
    return from_schema(draw(schemas), name="t")


def _numeric_cols(df):
    return [k for k, v in df.schema.items() if is_numeric(v)]


@st.composite
def chains(draw):
    """A random well-typed verb chain over a random schema."""
    df = draw(frames())
    for step in range(draw(st.integers(0, 5))):
        choices = ["filter_n", "mutate", "select", "arrange", "distinct",
                   "slice", "summarize"]
        verb = draw(st.sampled_from(choices))
        nums = _numeric_cols(df)
        if verb == "filter_n" and nums:
            c = draw(st.sampled_from(nums))
            df = df.filter(col[c] > draw(st.integers(-5, 5)))
        elif verb == "mutate" and nums:
            c = draw(st.sampled_from(nums))
            df = df.mutate(**{f"mut{step}": col[c] * 2})
        elif verb == "select":
            keep = draw(st.lists(st.sampled_from(df.columns), min_size=1,
                                 max_size=len(df.columns), unique=True))
            df = df.select(*keep)
        elif verb == "arrange":
            c = draw(st.sampled_from(df.columns))
            df = df.arrange(desc(col[c]) if draw(st.booleans()) else col[c])
        elif verb == "distinct":
            df = df.distinct()
        elif verb == "slice":
            df = df.slice_head(draw(st.integers(0, 10)))
        elif verb == "summarize":
            key = draw(st.sampled_from(df.columns))
            aggs = {f"cnt{step}": n()}  # step-unique: never collides with keys
            if nums and (c := draw(st.sampled_from(df.columns))) in nums and c != key:
                aggs[f"avg{step}"] = col[c].mean()
            df = df.group_by(col[key]).summarize(**aggs)
    return df


@given(chains())
def test_inference_is_total_and_schema_nonempty(df):
    # Constructing the chain already ran validation; the schema must be
    # well-formed: ordered, non-empty, every dtype known.
    assert df.columns
    assert all(d in (INT64, FLOAT64, BOOL, STR) for d in df.schema.values())


@given(chains())
def test_plan_hash_is_deterministic(df):
    assert plan_hash(df.plan) == plan_hash(df.plan)
    assert len(plan_hash(df.plan)) == 16


@given(frames())
def test_filter_mutate_roundtrip_schema(df):
    nums = _numeric_cols(df)
    if not nums:
        return
    c = nums[0]
    out = df.filter(col[c] > 0).mutate(z=col[c] + 1.5)
    assert out.schema == {**df.schema, "z": FLOAT64}


@given(frames())
def test_group_summarize_schema_is_keys_plus_aggs(df):
    key = df.columns[0]
    out = df.group_by(col[key]).summarize(n=n())
    assert out.columns == [key, "n"]
    assert out.schema["n"] == INT64
