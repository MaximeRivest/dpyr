# dpyr

**dplyr for Python.** The tidyverse's verbs — `filter`, `mutate`, `group_by`,
`summarize`, joins, `across`, tidyselect — as Python method chains, executing
on [polars](https://pola.rs) or [duckdb](https://duckdb.org), with real IDE
autocompletion and semantics verified against dplyr itself.

```bash
pip install dpyr        # or: uv add dpyr
```

```python
from dpyr import read_parquet, col, n, desc

starwars = read_parquet("starwars.parquet")

(
    starwars
    .filter(col.height > 180, col.mass < 100)
    .mutate(bmi = col.mass / (col.height / 100) ** 2)
    .group_by(col.species)
    .summarize(
        n = n(),
        mean_bmi = col.bmi.mean(),
    )
    .arrange(desc(col.mean_bmi))
)
```

Evaluate that in a notebook and you see rows immediately. Typo a column name
and you get the error *on that line*, with a did-you-mean suggestion. Wrap
the same code in a pipeline and only `.collect()` at the end, and the whole
chain runs as one fused query with predicate pushdown. That combination —
**schema-eager, data-lazy, display-eager** — is the core design.

## Two backends, one semantics

```python
import duckdb
from dpyr import from_duckdb, from_polars, from_dict

df  = from_dict({"x": [1, 2, 3], "g": ["a", "a", "b"]})   # polars engine
con = duckdb.connect("warehouse.db")
tbl = from_duckdb(con, "events")                          # SQL pushdown
```

Identical chains produce identical results on both engines — enforced by a
Hypothesis fuzzer that runs random verb chains on both and compares
bit-for-bit, and by **differential tests against real dplyr**: every spec in
`tests/specs/` is executed by dplyr (via `oracle/run_specs.R`) to produce a
committed golden parquet, then replayed through dpyr on both backends. Where
R and the engines genuinely disagree, the decision is documented in
[docs/SEMANTICS.md](docs/SEMANTICS.md), not left to chance.

## The dplyr you know

| dplyr | dpyr |
|---|---|
| `filter(df, height > 180)` | `df.filter(col.height > 180)` |
| `mutate(df, bmi = mass / h^2)` | `df.mutate(bmi = col.mass / col.h ** 2)` |
| `summarise(df, n = n(), m = mean(x, na.rm = TRUE))` | `df.summarize(n = n(), m = col.x.mean())` |
| `arrange(df, desc(mass))` | `df.arrange(desc(col.mass))` |
| `select(df, name, starts_with("h"))` | `df.select(col.name, starts_with("h"))` |
| `select(df, -mass)` | `df.select(-col.mass)` |
| `across(where(is.numeric), mean)` | `across(where(is_numeric), "mean")` |
| `left_join(a, b, by = "k")` | `a.left_join(b, on = col.k)` |
| `pivot_longer(df, x:y)` | `df.pivot_longer([col.x, col.y])` |
| `if_else()`, `case_when()`, `n_distinct()` | `if_else()`, `case_when()`, `.n_unique()` |
| `lag()`, `lead()`, `row_number()`, `min_rank()` | `lag()`, `lead()`, `row_number()`, `min_rank()` |
| `cumsum()`, `dense_rank()`, `percent_rank()` | `cum_sum()`, `dense_rank()`, `percent_rank()` |
| `slice_min(x, n)`, `slice_max(x, n)` (ties kept) | `slice_min(col.x, n)`, `slice_max(col.x, n)` |
| `separate()`, `unite()`, `relocate()` | `separate()`, `unite()`, `relocate()` |
| `coalesce()`, `replace_na()` | `coalesce()`, `replace_na()` |

Grouped `mutate`/`filter` are windowed per group, `summarize` peels one
grouping level, joins use `.x`/`.y` suffixes and match NAs by default —
the dplyr behaviors, deliberately.

## Autocompletion that actually works

- `df.c.height` — frame-bound proxy: column names complete from the live
  schema, and the returned expression is *typed* (`.mean()` on numerics,
  `.str_detect()` on strings; calling `.mean()` on a string column raises
  immediately, at build time).
- `df.filter(lambda c: c.height > 180)` — lambda style for the same effect.
- `dpyr stubgen data/*.parquet -o schemas.py` — generates typed schema
  modules so completion and type-checking work statically in any IDE.

## The database is a destination, not just a source

```python
db = dpyr.read_duckdb("warehouse.db")     # catalog object: db.tables, db.orders
gold = db.orders.group_by(col.region).summarize(rev = col.amount.sum())
gold.to_table("gold_revenue")             # CREATE TABLE AS <sql>, fully in-engine
gold.to_view("gold_live")                 # the lazy plan as a named view
gold.write_parquet("gold.parquet")        # in-engine COPY
mem = from_dict({"region": ["east"], "target": [1000.0]})
gold.inner_join(mem, on = col.region)     # in-memory frames bridge into duckdb
                                          # automatically (arrow, zero-copy)
```

## Interactive by default, lazy when you need it

```python
df.persist()           # checkpoint: materialize now (duckdb: temp table)
df.lazy()              # this frame never executes implicitly
dpyr.options.interactive = False   # global opt-out for production pipelines
```

Results are cached by plan hash, so re-displaying a frame in a notebook
never recomputes it.

## Documentation

Full guides at **[maximerivest.github.io/dpyr](https://maximerivest.github.io/dpyr/)** —
get started, grouped data, joins, window functions, column-wise operations,
reshaping, expressions & autocompletion, and the backends guide
(connecting and operating polars and duckdb).

## Project documents

| Doc | What it pins down |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | API design, the materialization model, autocompletion strategy, architecture |
| [docs/SEMANTICS.md](docs/SEMANTICS.md) | Every deliberate decision where R, polars and duckdb disagree |
| [docs/TESTING.md](docs/TESTING.md) | dplyr-as-oracle goldens, backend-agreement fuzzing, Hypothesis properties |
| [docs/ROADMAP.md](docs/ROADMAP.md) | What shipped in 1.0 and what's next |

## License

MIT © Maxime Rivest
