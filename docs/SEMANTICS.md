# SEMANTICS — the conformance spec

Where R/dplyr, polars and duckdb disagree, this file records the decision.
Every row below must be encoded as a test that links back here. Comparison
against the dplyr oracle is checked *modulo these documented divergences* —
never fuzzily.

Legend: **R** = follow dplyr · **P** = follow polars/duckdb · **pinned** =
our own rule, backends forced to comply.

| # | Area | dplyr | polars/duckdb | Decision |
|---|------|-------|---------------|----------|
| S1 | Missing values | typed `NA`, `NaN` distinct | `null` vs `NaN` | **pinned**: NA ↔ null bidirectionally; NaN preserved as NaN; document |
| S2 | `mean/sum/...` with missing | `NA` unless `na.rm=TRUE` | ignore nulls | **P**, with `na_rm: bool = True` kwarg for familiarity |
| S3 | Sort: NA position & stability | NAs last, stable sort | varies per engine | **pinned**: stable, NAs last; `desc()` keeps NAs last |
| S4 | `int / int` | promotes to double | varies | **R** (saner) |
| S5 | Integer overflow | promotes / warns | wraps or errors | **pinned**: Int64 default; overflow errors |
| S6 | String ordering / collation | locale-dependent (!) | byte/UTF-8 | **pinned**: C-locale codepoint order — *known divergence from R*; oracle harness normalizes |
| S7 | Grouped result ordering | sorted by group keys | hash order | **R**: sort by keys |
| S8 | Empty groups / zero-row inputs | specific dplyr behaviors | varies | **R**; port dplyr regression tests |
| S9 | `summarize` ungrouping | drops last group level | n/a | **R**, including the multi-key behavior |
| S10 | Join key NA matching | `NA` matches `NA` by default | SQL: NULL ≠ NULL | **R** default, `na_matches="never"` opt-out (mirrors dplyr arg) |
| S11 | Join suffixes | `.x` / `.y` | `_right` etc. | **R**: `(".x", ".y")` |
| S12 | Boolean with NA (3-valued logic) | NA propagates; `filter` drops NA | same in SQL | **R/SQL** (they agree); test it anyway |
| S13 | `n()` / counts dtype | integer | u32/i64 | **pinned**: Int64 |
| S14 | Division by zero | `Inf`/`NaN` | varies (duckdb errors on int) | **R**: `Inf`/`-Inf`/`NaN`, cast first on duckdb |
| S15 | `case_when` no match | `NA` | null | agree; pin the result dtype unification rule |
| S16 | Date/time zones | rich, messy | UTC-leaning | **pinned**: tz-aware UTC default; naive allowed; conversions explicit |
| S17 | Factors | core R type | none | not supported; oracle harness converts factors → strings before compare |
| S18 | Recycling length-1 values in `mutate` | yes | literals broadcast | **R** for scalars only; no general recycling |
| S19 | Float comparison in tests | — | — | harness: sort-normalize where order unspecified + ULP tolerance |

Process: when a differential test fails and the cause is a *new* semantic
disagreement, the fix is (1) add a row here, (2) encode it in the harness
normalization or backend compiler, (3) add a dedicated test naming the row.

Rows added during implementation (discovered by the oracle/fuzzer):

| # | Area | dplyr | polars/duckdb | Decision |
|---|------|-------|---------------|----------|
| S20 | `mean`/`median` of zero values (`na_rm=TRUE`, all missing) | `NaN` | `null` | **P**: null; oracle harness compares NaN==null for floats |
| S21 | Row order after joins, `pivot_longer`, `distinct`-dedup | left-order preserved | engine-dependent | **pinned**: unspecified; pin with `arrange()`; oracle/agreement tests sort before compare |
| S22 | Join output column positions | left columns keep their original positions, suffixed in place | varies | **R** (verified by join goldens) |
| S23 | `%` modulo | floor-mod (R `%%`) | polars floor-mod; SQL trunc-mod | **R**: duckdb compiles to `((a % b) + b) % b` |
| S24 | `is_in` with missing left value | `NA %in% xs` is FALSE | null propagates | **P**: null (filter drops it); divergence from R documented |
| S25 | grouped `slice_head` column order | original order | polars moves keys first | **R**: compiler restores schema order |
| S26 | `pivot_wider` duplicate keys | warns, builds list-columns | polars keeps first | **pinned**: UserWarning + keep first; a NULL in names_from becomes a column named 'null' (dplyr: 'NA') |
| S27 | duckdb sources from different connections in one plan | n/a | undefined | **pinned**: BackendError at collect; persist one side first |
| S28 | `arrange` before order-sensitive aggregates (`first`/`last`) | honored | engines vary | **R**: both backends honor pending sort order (ordered aggregates / framed windows on duckdb) |
| S29 | cumulative aggregates over missing values | NA poisons the rest (R cumsum) | polars: null at the null row, running total continues; SQL: skips | **P** (polars): null at the null position, total continues; divergence from R documented |
| S30 | `percent_rank` | (min_rank-1)/(non-missing-1) | SQL percent_rank counts NULL rows | **R**: built from min_rank and non-missing count on both backends |
| S31 | `separate()` sep | regex, default non-alnum | n/a | **pinned**: literal separator string, default "_" |
| S32 | `unite()` missing values | "NA" string (na.rm=FALSE) / dropped (TRUE) | engines skip or null | **R**: na_rm=False renders 'NA', na_rm=True drops (all-missing rows join to '') |
| S33 | `slice_sample` algorithm | R PRNG (`sample()`) | engine-native samplers differ | **pinned**: shared LCG-mix ordering — same seed selects the same rows on BOTH engines; differs from R's sampler by construction |
| S34 | mixing in-memory dataframes with duckdb tables | n/a | n/a | **pinned**: plans bridge automatically — duckdb scans the arrow data in place (zero-copy); two different duckdb connections still raise |
| S27 (rev. 1.7.0) | tables on different duckdb connections in one plan | n/a | undefined | **pinned**: the foreign side streams through arrow onto the primary connection (a copy, UserWarning emitted); to_view across connections still raises |
| S35 | list/array/struct columns | list-columns are first-class | engines hold them; comparison semantics (nulls inside, ordering) differ | **pinned** (1.9.0): nested columns are *carried* — select/filter by other columns/mutate copies/lag/lead/first/last/is_na/joins on other keys/persist/parquet·jsonl·arrow writes — and never *compared*: sorting, grouping, dedup, join keys, pivots, comparisons and aggregates other than first/last on them raise at build time. Dtypes are named canonically (`List(Str)`, `Struct(a: Int64)`, `Array(Float64, 3)`; inner ints/floats widened) so both engines agree. CSV/TSV/Excel writes refuse them (polars refuses, duckdb stringifies) |
| S36 | `read(list of records)` | `bind_rows()` of rows | polars samples the first 100 rows (drops late keys, truncates late floats in lists, stringifies int+str) | **R**: union of keys in first-seen order, missing → NA; every row is scanned (built through arrow); bool+number → number and date+datetime → datetime (vctrs); any other mix raises naming the column and rows; an empty list is a 0x0 table (`bind_rows(list())`) (1.11.0) |
| S37 | `mean`/`sd`/`var` of a logical | TRUE/FALSE count as 1/0 (`mean(correct)` is the share of TRUE) | polars casts; duckdb has no `avg(BOOLEAN)` | **R** (1.9.0): both backends average the 0/1 values; `median` of a logical stays an error |
| S38 | Python functions per row (`vectorize`) | `mutate(y = map_chr(x, f))`, `pmap()` for several inputs | polars `map_elements`, duckdb Python UDFs; neither is shared, both lose the type at build time | **pinned** (1.10.0): the frame layer lifts each call into a RowMap step that runs in Python between engine steps, so both engines agree by construction. The column type comes from the return annotation (checked at build time). One call per distinct argument set, remembered per (function, version) for the session. Display pushes `head()` below row functions through row-local steps only. Failures raise after all rows ran (purrr aborts at the first); `errors="null"` keeps going. Valid in `mutate()`/`filter()` only |
