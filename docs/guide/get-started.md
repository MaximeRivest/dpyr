# Get started

dpyr gives you dplyr's verb vocabulary — `filter`, `mutate`, `group_by`,
`summarize`, and friends — as Python method chains that execute on
[polars](https://pola.rs) or [duckdb](https://duckdb.org). You don't need to
know any R: if you've used pandas or polars, the verbs read as plain English
and the rest of this page shows you everything by example.

## Install

```bash
pip install dpyr
# or
uv add dpyr
```

## A dataset to play with

We'll build a frame from plain Python lists so this page is fully
self-contained — but `read()` takes far more than dicts: parquet, CSV,
JSON, Excel, duckdb and SQLite files, URLs, pandas frames, Hugging Face
datasets... the whole tour is in [Reading & writing](reading/index.md). Here's
a season of harvest records from a small market garden — one row per
crop picked from a bed:

```python
from dpyr import read, col, n, desc, starts_with

harvest = read({
    "crop": ["tomato", "tomato", "zucchini", "zucchini",
             "kale", "kale", "garlic", "garlic", "carrot"],
    "bed":  ["A1", "B2", "A1", "C3", "B2", "C3", "A1", "B2", "C3"],
    "kg":   [12.4, 9.1, 18.0, 22.5, 3.2, None, 4.0, 4.4, 7.5],
    "price_per_kg":     [5.50, 5.50, 3.00, 3.00, 8.00, 8.00, 12.00, 12.00, 4.25],
    "days_to_maturity": [78, 81, 52, 55, 60, 62, 240, 240, 70],
})
print(harvest)
```

```text
# dpyr frame · source: polars · showing 9 of 9 rows
shape: (9, 5)
┌──────────┬─────┬──────┬──────────────┬──────────────────┐
│ crop     ┆ bed ┆ kg   ┆ price_per_kg ┆ days_to_maturity │
│ ---      ┆ --- ┆ ---  ┆ ---          ┆ ---              │
│ str      ┆ str ┆ f64  ┆ f64          ┆ i64              │
╞══════════╪═════╪══════╪══════════════╪══════════════════╡
│ tomato   ┆ A1  ┆ 12.4 ┆ 5.5          ┆ 78               │
│ tomato   ┆ B2  ┆ 9.1  ┆ 5.5          ┆ 81               │
│ zucchini ┆ A1  ┆ 18.0 ┆ 3.0          ┆ 52               │
│ zucchini ┆ C3  ┆ 22.5 ┆ 3.0          ┆ 55               │
│ kale     ┆ B2  ┆ 3.2  ┆ 8.0          ┆ 60               │
│ kale     ┆ C3  ┆ null ┆ 8.0          ┆ 62               │
│ garlic   ┆ A1  ┆ 4.0  ┆ 12.0         ┆ 240              │
│ garlic   ┆ B2  ┆ 4.4  ┆ 12.0         ┆ 240              │
│ carrot   ┆ C3  ┆ 7.5  ┆ 4.25         ┆ 70               │
└──────────┴─────┴──────┴──────────────┴──────────────────┘
```

Two things to notice already:

- **Evaluating a frame shows rows.** In a notebook or REPL, the repr runs the
  plan and prints a preview — no `.collect()` ceremony while you explore.
  Under the hood frames stay lazy and results are cached by plan hash, so
  re-displaying never recomputes.
- **`col.kg` refers to a column.** `col` is a free-floating column proxy, the
  equivalent of bare column names in dplyr or `pl.col(...)` in polars. The
  frame-bound flavor `harvest.c.kg` does the same job but autocompletes from
  the live schema in your editor.

One kale row has `kg = None` — that becomes a missing value (`null`), which
several verbs below treat specially.

## Errors arrive immediately, with a suggestion

Because every frame knows its schema eagerly, a typo'd column name fails on
the line that made it — not three cells later inside an engine traceback:

```python
try:
    harvest.filter(col.priced_per_kg > 4)   # oops: priced_per_kg
except Exception as e:
    print(f"{type(e).__name__}: {e}")
```

```text
ColumnNotFoundError: column 'priced_per_kg' not found in filter(). Did you mean 'price_per_kg'? Available columns: crop, bed, kg, price_per_kg, days_to_maturity
```

## filter — keep rows

Pass several conditions and they are AND-ed together, like dplyr (use `|` for
OR within one expression):

```python
print(harvest.filter(col.kg > 5, col.price_per_kg < 6))
```

```text
# dpyr frame · source: polars · showing 5 of 5 rows
shape: (5, 5)
┌──────────┬─────┬──────┬──────────────┬──────────────────┐
│ crop     ┆ bed ┆ kg   ┆ price_per_kg ┆ days_to_maturity │
│ ---      ┆ --- ┆ ---  ┆ ---          ┆ ---              │
│ str      ┆ str ┆ f64  ┆ f64          ┆ i64              │
╞══════════╪═════╪══════╪══════════════╪══════════════════╡
│ tomato   ┆ A1  ┆ 12.4 ┆ 5.5          ┆ 78               │
│ tomato   ┆ B2  ┆ 9.1  ┆ 5.5          ┆ 81               │
│ zucchini ┆ A1  ┆ 18.0 ┆ 3.0          ┆ 52               │
│ zucchini ┆ C3  ┆ 22.5 ┆ 3.0          ┆ 55               │
│ carrot   ┆ C3  ┆ 7.5  ┆ 4.25         ┆ 70               │
└──────────┴─────┴──────┴──────────────┴──────────────────┘
```

The kale row with the missing weight is gone too: `null > 5` evaluates to
`null`, and `filter` only keeps rows where the predicate is *true* — the
dplyr/SQL three-valued-logic rule (see SEMANTICS S12). To target missing
values explicitly use `.is_na()` — `harvest.filter(col.kg.is_na())` returns
just that kale row, and `harvest.filter(~col.kg.is_na())` its complement.

## arrange — sort rows

`desc()` flips a key to descending. Sorting is stable, and missing values go
last in both directions (SEMANTICS S3) — note where the null-weight kale lands:

```python
print(harvest.arrange(desc(col.kg)).slice_tail(3))
```

```text
# dpyr frame · source: polars · showing 3 of 3 rows
shape: (3, 5)
┌────────┬─────┬──────┬──────────────┬──────────────────┐
│ crop   ┆ bed ┆ kg   ┆ price_per_kg ┆ days_to_maturity │
│ ---    ┆ --- ┆ ---  ┆ ---          ┆ ---              │
│ str    ┆ str ┆ f64  ┆ f64          ┆ i64              │
╞════════╪═════╪══════╪══════════════╪══════════════════╡
│ garlic ┆ A1  ┆ 4.0  ┆ 12.0         ┆ 240              │
│ kale   ┆ B2  ┆ 3.2  ┆ 8.0          ┆ 60               │
│ kale   ┆ C3  ┆ null ┆ 8.0          ┆ 62               │
└────────┴─────┴──────┴──────────────┴──────────────────┘
```

## The slice family — take rows by position, rank, or chance

`slice_head(n)` / `slice_tail(n)` take the first/last *n* rows.
`slice_sample(n, seed=...)` draws rows at random — pass a seed and the draw
is fully reproducible: dpyr uses one shared sampling algorithm, so the same
seed selects the **same rows on both engines** (SEMANTICS S33), and the
result is safe to cache because the seed lives in the query plan rather
than in hidden RNG state. `slice_min` / `slice_max` take the rows with
the extreme values of an expression, **keeping ties** like dplyr does:

```python
print(harvest.slice_max(col.days_to_maturity, n=1))   # garlic ties at 240
```

```text
# dpyr frame · source: polars · showing 2 of 2 rows
shape: (2, 5)
┌────────┬─────┬─────┬──────────────┬──────────────────┐
│ crop   ┆ bed ┆ kg  ┆ price_per_kg ┆ days_to_maturity │
│ ---    ┆ --- ┆ --- ┆ ---          ┆ ---              │
│ str    ┆ str ┆ f64 ┆ f64          ┆ i64              │
╞════════╪═════╪═════╪══════════════╪══════════════════╡
│ garlic ┆ A1  ┆ 4.0 ┆ 12.0         ┆ 240              │
│ garlic ┆ B2  ┆ 4.4 ┆ 12.0         ┆ 240              │
└────────┴─────┴─────┴──────────────┴──────────────────┘
```

You asked for one row and got two — pass `with_ties=False` for exactly *n*.

## select and rename — pick and relabel columns

`select` accepts column refs, tidyselect helpers (`starts_with`, `contains`,
`where(is_numeric)`, ...), and negations to drop:

```python
print(harvest.select(col.crop, starts_with("k")).schema)   # crop, kg
print(harvest.select(-col.days_to_maturity).schema)        # drop one column
```

```text
{'crop': Str, 'kg': Float64}
{'crop': Str, 'bed': Str, 'kg': Float64, 'price_per_kg': Float64}
```

`rename` uses dplyr's `new = old` direction, keywords on the left:

```python
print(harvest.rename(weight_kg=col.kg).schema)
```

```text
{'crop': Str, 'bed': Str, 'weight_kg': Float64, 'price_per_kg': Float64, 'days_to_maturity': Int64}
```

## mutate — add columns

New columns are computed in order, so a later expression can use one defined
earlier *in the same call* — no intermediate assignment needed:

```python
priced = harvest.mutate(
    revenue = col.kg * col.price_per_kg,
    revenue_per_day = col.revenue / col.days_to_maturity,   # uses revenue
)
print(priced.select(col.crop, col.bed, col.revenue, col.revenue_per_day).slice_head(3))
```

```text
# dpyr frame · source: polars · showing 3 of 3 rows
shape: (3, 4)
┌──────────┬─────┬─────────┬─────────────────┐
│ crop     ┆ bed ┆ revenue ┆ revenue_per_day │
│ ---      ┆ --- ┆ ---     ┆ ---             │
│ str      ┆ str ┆ f64     ┆ f64             │
╞══════════╪═════╪═════════╪═════════════════╡
│ tomato   ┆ A1  ┆ 68.2    ┆ 0.874359        │
│ tomato   ┆ B2  ┆ 50.05   ┆ 0.617901        │
│ zucchini ┆ A1  ┆ 54.0    ┆ 1.038462        │
└──────────┴─────┴─────────┴─────────────────┘
```

Missing values propagate through arithmetic: the kale row with no weight gets
a `null` revenue, not an error.

## group_by + summarize — collapse groups to rows

`n()` counts rows per group; aggregates like `.sum()` and `.mean()` skip
missing values by default (`na_rm=True`, matching the habit of dplyr users —
SEMANTICS S2). Group keys come back sorted (S7), and `summarize` returns an
ungrouped frame when one grouping level remains (S9):

```python
print(
    priced
    .group_by(col.crop)
    .summarize(
        picks = n(),
        total_kg = col.kg.sum(),
        avg_revenue = col.revenue.mean(),
    )
    .arrange(desc(col.total_kg))
)
```

```text
# dpyr frame · source: polars · showing 5 of 5 rows
shape: (5, 4)
┌──────────┬───────┬──────────┬─────────────┐
│ crop     ┆ picks ┆ total_kg ┆ avg_revenue │
│ ---      ┆ ---   ┆ ---      ┆ ---         │
│ str      ┆ i64   ┆ f64      ┆ f64         │
╞══════════╪═══════╪══════════╪═════════════╡
│ zucchini ┆ 2     ┆ 40.5     ┆ 60.75       │
│ tomato   ┆ 2     ┆ 21.5     ┆ 59.125      │
│ garlic   ┆ 2     ┆ 8.4      ┆ 50.4        │
│ carrot   ┆ 1     ┆ 7.5      ┆ 31.875      │
│ kale     ┆ 2     ┆ 3.2      ┆ 25.6        │
└──────────┴───────┴──────────┴─────────────┘
```

`count()` is the usual shorthand for "group and tally":

```python
print(harvest.count(col.bed))
```

```text
# dpyr frame · source: polars · showing 3 of 3 rows
shape: (3, 2)
┌─────┬─────┐
│ bed ┆ n   │
│ --- ┆ --- │
│ str ┆ i64 │
╞═════╪═════╡
│ A1  ┆ 3   │
│ B2  ┆ 3   │
│ C3  ┆ 3   │
└─────┴─────┘
```

## The same verbs on duckdb

Point a frame at a duckdb table and the identical chain compiles to SQL with
full pushdown — same verbs, same results (that agreement is enforced by
differential tests against dplyr itself):

```python
import duckdb
from dpyr import read

con = duckdb.connect()   # in-memory database
con.execute("""
    CREATE TABLE picks AS SELECT * FROM (VALUES
        ('tomato', 12.4), ('tomato', 9.1),
        ('zucchini', 18.0), ('zucchini', 22.5),
        ('kale', 3.2), ('kale', NULL)
    ) AS t(crop, kg)
""")

print(
    read(con, "picks")
    .group_by(col.crop)
    .summarize(picks=n(), total_kg=col.kg.sum())
    .arrange(desc(col.total_kg))
)
```

```text
# dpyr frame · source: duckdb · showing 3 of 3 rows
shape: (3, 3)
┌──────────┬───────┬──────────┐
│ crop     ┆ picks ┆ total_kg │
│ ---      ┆ ---   ┆ ---      │
│ str      ┆ i64   ┆ f64      │
╞══════════╪═══════╪══════════╡
│ zucchini ┆ 2     ┆ 40.5     │
│ tomato   ┆ 2     ┆ 21.5     │
│ kale     ┆ 2     ┆ 3.2      │
└──────────┴───────┴──────────┘
```

When you're done exploring, `.collect()` returns a polars `DataFrame`
(`.to_pandas()` for pandas), and `.lazy()` / `options.interactive = False`
turn off implicit execution for production pipelines. Collecting isn't the
only way out, though: `df.write("results.parquet")`, `to_table()`, and friends land
results straight from the engine, and you can even mix in-memory frames with
duckdb tables in one plan — see [Backends](backends.md) for the details.

## glimpse — the quick look

For a fast "what am I holding?" check, `glimpse()` prints one line per
column — dtype plus the leading values — which fits wide tables on screen
far better than a row-oriented preview:

```python
harvest.glimpse()
```

```text
Rows: 9
Columns: 5
$ crop             <Str> 'tomato', 'tomato', 'zucchini', 'zucchini', 'kale', 'kale', 'garlic', 'garlic', 'carrot'
$ bed              <Str> 'A1', 'B2', 'A1', 'C3', 'B2', 'C3', 'A1', 'B2', 'C3'
$ kg               <Float64> 12.4, 9.1, 18.0, 22.5, 3.2, NA, 4.0, 4.4, 7.5
$ price_per_kg     <Float64> 5.5, 5.5, 3.0, 3.0, 8.0, 8.0, 12.0, 12.0, 4.25
$ days_to_maturity <Int64> 78, 81, 52, 55, 60, 62, 240, 240, 70
```

It returns the frame, so it slots into the middle of a chain while you
debug.

## Where next

- [Grouped data](grouped-data.md) — grouped `mutate`/`filter`, multi-key
  `summarize`, slicing within groups
- [Joins](joins.md) — `left_join` and friends, `.x`/`.y` suffixes, NA-key
  matching
- [Window functions](window-functions.md) — `lag`, `lead`, rankings,
  cumulative aggregates
- [Column-wise operations](column-wise.md) — `across()` with tidyselect
- [Reshaping](reshaping.md) — `pivot_longer`/`pivot_wider`, `separate`,
  `unite`
- [Reading & writing](reading/index.md) — every format and source `read()`
  and `write()` understand
- [Backends](backends.md) — polars vs duckdb, `persist()`, lazy mode,
  caching, and ML data (Hugging Face datasets, numpy, torch, jax)
