# ROADMAP — epics & stories to MVP

Ordered by dependency. A story is done when its tests (per TESTING.md) are
green on both backends. MVP = Epics 0–7 complete.

**MVP definition:** a user can `uv add dpyr`, read parquet/CSV or a
duckdb table, run the core dplyr verb set with the `col` proxy, get
immediate schema errors and auto-displayed results in Jupyter, and trust
the output because it's differentially tested against dplyr.

---

## Epic 0 — Project foundation
- **0.1** Repo scaffold: uv project, `pyproject.toml`, ruff, mypy,
  pytest, GitHub Actions. ✅
- **0.2** ✅ Reserve `dpyr` on PyPI with a 0.0.1 placeholder that
  errors helpfully on import-and-use. (`dataframe` is PyPI policy-blocked;
  see DESIGN.md §6 for the naming history.)
- **0.3** Commit intent docs (DESIGN/SEMANTICS/TESTING/ROADMAP). ✅

## Epic 1 — Expression IR & schema engine (the core) ✅
- **1.1** ✅ Dtype system: Int64/Float64/Bool/Str/Date/Datetime/Null + NA
  model (SEMANTICS S1).
- **1.2** ✅ Expression nodes: column refs, literals, arithmetic/comparison/
  boolean ops, function calls; typed expr classes (NumExpr, StrExpr, ...).
- **1.3** ✅ `col` proxy producing typed expressions; `desc()`, `n()`,
  `if_else()`, `case_when()`.
- **1.4** ✅ Logical plan nodes for every MVP verb; **schema inference +
  validation per node** with did-you-mean errors and one-frame tracebacks.
- **1.5** ✅ Plan hashing (for the materialization cache) and stable repr
  (for IR snapshot tests).
- Tests: unit + Hypothesis property 1 (schema soundness).

## Epic 2 — polars backend ✅
- **2.1** Sources: `read_parquet`, `read_csv`, `from_polars`, `from_pandas`.
- **2.2** Compile IR → polars LazyFrame for: filter, mutate, select,
  rename, arrange, distinct, slices.
- **2.3** group_by/summarize incl. grouped ordering (S7) and ungroup
  semantics (S9).
- **2.4** Joins (all six) with dplyr suffix & NA-matching rules (S10–S11).
- **2.5** Semantics shims: sort stability/NA position (S3), int division
  (S4), counts dtype (S13), division by zero (S14).

## Epic 3 — Materialization model (the UX) ✅
- **3.1** Display-eager boundaries: `__repr__`/`_repr_html_` with capped
  preview + provenance line, `len`, `.shape`, iteration, `.to_polars()`,
  `.to_pandas()`, `.pull()`.
- **3.2** Result cache keyed by plan hash; invalidation on `persist()`.
- **3.3** `.persist()` checkpoint verb; `.lazy()` / `mode="lazy"` opt-out.
- **3.4** Implicit persist for schema-needs-data ops (groundwork for
  pivot_wider).
- Tests: metamorphic laws on persist/collect/repr equivalence.

## Epic 4 — Oracle harness (start early, runs forever) ✅
- **4.1** YAML spec format + Python spec runner.
- **4.2** `oracle/run_specs.R` + pinned rocker container; golden parquet
  generation; fixture metadata (dplyr version).
- **4.3** Comparison/normalization harness implementing SEMANTICS
  S6/S17/S19.
- **4.4** CI: per-push spec tests vs committed goldens; weekly golden
  drift job (semantic comparison in a pinned rocker container).
- **4.5** Seed corpus: 42 hand-written specs across all MVP verbs
  (filter is deepest at 8), derived from dplyr doc examples.

## Epic 5 — duckdb backend ✅
- **5.1** Sources: `from_duckdb(con, table)`, `read_sql`.
- **5.2** IR → SQL compiler for the Epic-2 verb set, with semantics shims
  (casts, ORDER BY NULLS LAST, etc.).
- **5.3** `persist()` as `CREATE TEMP TABLE`.
- **5.4** Hypothesis property 2: backend agreement, in CI on every push.

## Epic 6 — tidyselect, across, reshaping ✅
- **6.1** tidyselect: `starts_with/ends_with/contains/matches/where/
  everything`, negation; works in `select`, `rename`, `distinct`.
- **6.2** `across()` in mutate/summarize with `names=` templating.
- **6.3** `pivot_longer`; `pivot_wider` (uses 3.4); `count`; `pull`.
- **6.4** `GroupedDFrame` type with its own completion surface; grouped
  filter/mutate window semantics.

## Epic 7 — Autocompletion & developer experience ✅
- **7.1** Runtime completion: `__dir__` from live schema; frame-bound
  `df.c` proxy.
- **7.2** `DFrame[S]` generic typing + lambda-style `filter(lambda c: ...)`.
- **7.3** `dpyr stubgen` CLI: parquet/duckdb schema → `Cols` stubs;
  stubs checked by mypy+pyright in CI.
- **7.4** Error message polish pass (the "feels eager" acceptance test:
  every user mistake surfaces on the line that made it).

## Epic 8 — Hardening & release (MVP gate) ✅ (1.0.0 published 2026-06-10)
- **8.1** Regression coverage for MVP verbs via the oracle corpus,
  the backend-agreement fuzzer, and the adversarial-review regression
  suite (tests/test_review_regressions.py).
- **8.2** Nightly backend-agreement fuzzing job (raised example count).
- **8.3** Docs for 1.0: README tutorial + dplyr→dpyr cheat sheet;
  SEMANTICS/DESIGN/TESTING in-repo. (Dedicated site: post-1.0.)
- **8.4** `1.0.0` to PyPI; announce with the differential-test count as
  the headline ("passes N dplyr-generated golden tests").

## 1.1.0 ✅ (published 2026-06-10)
Window functions (`lag`/`lead`/`row_number`/`min_rank`/`dense_rank`/
`percent_rank`/`cum_sum`/`cum_min`/`cum_max`), `slice_min`/`slice_max`
with dplyr tie semantics, `separate`/`unite`/`relocate`,
`coalesce`/`replace_na` — on both backends, oracle-tested (50 specs) and
fuzz-tested. Fixed a synthesized-SQL name-collision bug that caused rare
nondeterministic slice results on duckdb (S3).

## 1.2.0 ✅ (published 2026-06-10)
The engine disappears: in-memory frames bridge into duckdb automatically
(arrow scanned in place, zero-copy) so mixing is no longer an error;
collect(engine=) override; persist()/to_table()/to_view() materialize
fully in-engine (CREATE ... AS <sql>); write_parquet via in-engine COPY;
readr-style IO (read_duckdb catalog object, read_ipc memory-mapped,
write_duckdb, glimpse()); slice_sample unified across engines (S33).

## 1.3.0 ✅ (published 2026-06-10)
One reader, one writer: read(path) and df.write(path) dispatch on file
extension (.parquet/.pq, .csv, .arrow/.feather/.ipc, .db/.duckdb/.ddb —
duckdb files open as a catalog, write to a named table). write_csv added
(in-engine COPY on duckdb). Format-specific functions remain as explicit
escape hatches.

## Post-MVP (parking lot)
`nest`,
list-columns, streaming collect, arrow Flight sources, sqlite/postgres
backends via the duckdb SQL layer, plugin API for custom verbs.
