# TESTING — strategy

Three layers. dplyr is the oracle; polars↔duckdb agreement is the
workhorse; Hypothesis finds what nobody thought to write.

## Layer 1 — differential testing against dplyr (the oracle)

Test cases live in a neutral spec format (YAML), input data + verb chain:

```yaml
# tests/specs/summarize/grouped_mean.yaml
input: starwars
chain:
  - filter: "height > 180"
  - group_by: [species]
  - summarize: {n: "n()", mh: "mean(height, na.rm=TRUE)"}
```

- `oracle/run_specs.R` translates each spec to dplyr code, runs it, writes
  the result as **parquet** (lossless types, NA fidelity — never CSV) into
  `tests/golden/`.
- pytest runs the same spec through our library (both backends) and
  asserts dataframe equality via the normalization harness (see SEMANTICS.md
  S6/S17/S19).

**R stays out of the inner loop.** Golden parquets are generated offline
and committed. Day-to-day `pytest` is pure Python. A separate CI job in a
pinned `rocker/tidyverse` container regenerates all goldens and fails on
drift — catching both our bugs and wrongly-encoded expectations. dplyr
version is pinned and recorded in fixture metadata.

## Layer 2 — mined test corpus

- Port **dplyr's documented examples** (every verb's examples, vignettes,
  relevant R4DS chapters) into specs. They double as our tutorial docs.
- Read **dplyr's testthat suite** and port its *regression tests* — each
  is a documented historical bug (zero-row groups, grouped filter, join
  multiplicities, `across()` corners).
- **dbplyr's translation tests** inform the duckdb compiler.

## Layer 3 — property-based testing (Hypothesis)

Strategies generate small random dataframes (mixed dtypes, nulls, empty,
single-row, duplicates) and type-correct random verb chains (a grammar
over the IR).

Properties, in order of cost:

1. **Schema soundness** (no oracle): predicted output schema ==
   actually-collected schema. Cheap; run always. This is the test of the
   schema-eager promise.
2. **Backend agreement** (no oracle): polars result == duckdb result for
   the same plan. Our most important invariant. Run always.
3. **Metamorphic laws** (no oracle):
   - `filter(p).filter(q)` ≡ `filter(p & q)`
   - `arrange ∘ filter` ≡ `filter ∘ arrange` (as multisets)
   - `mutate` of an unused column doesn't change downstream `summarize`
   - chain collected once ≡ collected verb-by-verb with `persist()`
     between every step ← *specifically exercises the lazy/eager machinery*
   - repr (display-eager) then collect ≡ collect directly (cache
     correctness)
4. **Oracle fuzzing** (needs R): sampled generated cases through the
   dplyr oracle. Nightly job, not per-commit.

## Unit layers (ordinary but required)

- IR construction & schema inference per verb (golden IR snapshots).
- duckdb compiler: IR → SQL string snapshots.
- Error UX: wrong column names raise immediately with short tracebacks
  and did-you-mean suggestions (asserted on message content).
- Stubgen: generated stubs compile under mypy & pyright in CI.

## CI matrix

| Job | Trigger | Needs R |
|---|---|---|
| pytest (specs vs committed goldens, both backends) | every push | no |
| Hypothesis schema/backend/metamorphic | every push | no |
| golden regeneration + drift check | weekly + manual dispatch | yes (docker) |
| backend-agreement fuzzing (raised examples) | nightly | no |
| mypy/pyright incl. generated stubs | every push | no |
