# DESIGN — intent and architecture

This document records the design decisions and *why* they were made, so the
project stays coherent as it grows. Changes to anything in here deserve a
discussion, not a drive-by PR.

## 1. The problem

dplyr's ergonomics rest on non-standard evaluation (bare column names),
which Python cannot do. Prior ports each gave up something essential:

- **dfply / dplython** — emulated `%>%` with `>>` operator hacks; alien to
  Python, broke tooling, unmaintained.
- **siuba** — closest in spirit (`_` proxy, SQL backend) but weak typing
  and completion.
- **tidypolars** — dplyr verb names over polars, but string column refs:
  no completion, no expression typing.
- **ibis** — the right architecture (expression IR, many backends, lazy
  with interactive mode) but the API drifted far from dplyr.

We take ibis's architecture, dplyr's API, and add the missing piece nobody
shipped: schema-aware autocompletion.

## 2. API surface

### Verbs (dplyr names, verbatim)

MVP set: `filter`, `mutate`, `select`, `rename`, `arrange`, `group_by`
(+ implicit ungroup after `summarize`), `summarize`/`summarise`,
`distinct`, `slice_head`/`slice_tail`/`slice_sample`, `count`,
`left_join`/`inner_join`/`right_join`/`full_join`/`semi_join`/`anti_join`,
`pivot_longer`, `pivot_wider`, `pull`.

Pipe = method chaining. No `>>`/`|` operator overloading, ever.
New columns via kwargs: `mutate(bmi = ...)`, `summarize(n = n())`.

### The `col` proxy

`col.height > 180` builds an expression tree (our IR), not a value.
Column-typed expression classes (`NumExpr`, `StrExpr`, `BoolExpr`,
`TemporalExpr`, ...) carry the appropriate methods (`.mean()`, `.str_detect()`,
`.year()`), so completion is type-correct.

Helpers as plain functions: `n()`, `desc()`, `if_else()`, `case_when()`,
`across()`, and tidyselect (`starts_with`, `ends_with`, `contains`,
`matches`, `where`, `everything`); negation via unary minus.

### Autocompletion strategy (the differentiator)

Three tiers, weakest to strongest:

1. **Runtime**: `DFrame.__dir__` and the frame-bound `df.c` proxy
   populated from the live schema → Jupyter/REPL completion for free.
2. **Generic typing**: `DFrame[S]` parameterized by a `Cols` schema class;
   `filter(lambda c: c.height > 180)` completes via the lambda's inferred
   parameter type.
3. **Stub codegen**: a CLI (`dpyr stubgen data/*.parquet`) reads
   parquet/duckdb schemas and emits `Cols` subclasses + typed module
   attributes, giving full static completion and type-checking in any IDE.

## 3. Materialization model

**Schema-eager, data-lazy, display-eager.** (The core UX decision.)

- *Schema-eager*: every verb validates inputs against the known schema and
  computes its output schema synchronously. Wrong column name, type
  mismatch, bad group reference → exception on that line, with a one-frame
  traceback. Requires only metadata; costs nothing.
- *Data-lazy*: verbs append to a logical plan. No intermediate
  materialization within a chain → query fusion, predicate pushdown into
  parquet/duckdb.
- *Display-eager* (default on): materialization happens automatically at
  the boundaries where a value escapes the expression world —
  `__repr__`/`_repr_html_` (fetch a capped preview, like tibble's 10-row
  print), `len`, `.shape`, iteration, `.to_polars()`, `.to_pandas()`,
  plotting hooks. In a notebook this *feels* fully eager; in a pipeline
  that only collects at the end, the same code gets full laziness.
  `mode="lazy"` / `.lazy()` opts out for production.

Rationale: dplyr rose to fame *eager*; immediate errors and immediate
results are why interactive analysis felt good. Immediate errors need only
the schema. Immediate results need collection only at display points.

### Sharp edges, handled deliberately

- **Repeated re-execution**: cache results on first materialization, keyed
  by plan hash. Plus an explicit `.persist()` checkpoint verb
  (polars: collected frame; duckdb: `CREATE TEMP TABLE`).
- **Source mutation between displays**: `persist()` is the snapshot
  operator; documented honestly.
- **Schema-needs-data ops** (`pivot_wider` — output columns come from
  values): implicitly persist their input, compute, continue. Users never
  see the distinction.
- **Provenance in print**: repr shows collected rows plus
  `# dpyr frame · source: polars · showing 10 of ? rows` (the total
  is exact once the result is cached; `(lazy)` appears when interactive
  display is off).

## 4. Architecture

```
user API (verbs, col proxy)
        │  builds
        ▼
logical plan + expression IR  ←— schema inference/validation lives here
        │  compiles to
        ├──────────────► polars LazyFrame (expression 1:1 mapping)
        └──────────────► SQL string / relation for duckdb
```

- The IR is *ours* and small; semantics (NA handling, sort stability,
  grouped ordering — see SEMANTICS.md) are pinned in the IR, and each
  backend compiler is responsible for complying, inserting casts/sorts
  where the engine's defaults differ.
- Backend chosen at the source: `read_parquet/read_csv/from_polars/
  from_pandas` → polars; `from_duckdb(con, "tbl")` / `read_sql` → duckdb.
- `group_by` returns `GroupedDFrame` (separate type → separate completion
  surface), auto-ungrouping after `summarize`, matching dplyr.

## 5. Non-goals (MVP)

- No pandas execution backend (only conversion in/out).
- No R-style NSE magic via frame inspection / AST tricks.
- No plotting, no modeling. Frames out are polars/pandas; the ecosystem
  does the rest.
- No distributed execution.

## 6. Name

PyPI name: `dpyr` — dplyr with the L dropped, so "py" sits in the middle:
dplyr-for-Python in four characters. Chosen 2026-06-09 after `dataframe`
turned out to be PyPI policy-blocked and `dataframes`/`table`/most
expressive English words were squatted; `tibble` and `dplyr` were free but
carry Posit brand-risk. Import name: `dpyr`.
