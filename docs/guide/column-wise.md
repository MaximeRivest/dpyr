# Column-wise operations

Sooner or later every pipeline hits the "do this to twenty columns" problem.
In pandas you reach for `df.filter(regex=...)` or a dict passed to `.agg`;
in polars, `pl.col("^bill_.*$")` or `cs.numeric()`. dpyr splits the problem
into two composable pieces, both borrowed from dplyr:

- **selectors** (`starts_with`, `where(is_numeric)`, ...) answer *which columns*,
- **`across()`** answers *what to do to each of them*.

Selectors work anywhere a column set is expected — `select()`, `relocate()`,
and inside `across()` — and they resolve against the frame's schema at the
moment the verb is called, so a bad pattern fails on that line, not at
`collect()` time.

Everything below is one runnable script. Setup:

```python
import duckdb
from dpyr import (
    read, col, n, across,
    starts_with, ends_with, contains, matches, where, everything,
    is_numeric, is_string,
)

penguins = read({
    "species":           ["Adelie", "Adelie", "Gentoo", "Gentoo", "Chinstrap", "Chinstrap"],
    "island":            ["Torgersen", "Dream", "Biscoe", "Biscoe", "Dream", "Dream"],
    "bill_length_mm":    [39.1, 37.8, 47.5, 49.9, 46.5, 51.3],
    "bill_depth_mm":     [18.7, 18.3, 15.0, 16.1, 17.9, 19.2],
    "flipper_length_mm": [181.0, 174.0, 217.0, 213.0, 192.0, 198.0],
    "year":              [2007, 2008, 2007, 2008, 2007, 2008],
})
print(penguins.columns)
```

```text
['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'year']
```

## Picking columns: tidyselect in `select()`

`select()` takes any mix of plain column references and selectors:

| Selector | Keeps columns whose... |
|---|---|
| `starts_with("bill")` | name starts with the prefix |
| `ends_with("_mm")` | name ends with the suffix |
| `contains("length")` | name contains the substring |
| `matches(r"^bill_.*_mm$")` | name matches the regex (Python `re.search`) |
| `where(is_numeric)` | *dtype* satisfies the predicate (`is_string`, `is_bool`, or your own `DType -> bool`) |
| `everything()` | always — useful for "and then the rest" |

```python
print(penguins.select(starts_with("bill")))
```

```text
# dpyr frame · source: polars · showing 6 of 6 rows
shape: (6, 2)
┌────────────────┬───────────────┐
│ bill_length_mm ┆ bill_depth_mm │
│ ---            ┆ ---           │
│ f64            ┆ f64           │
╞════════════════╪═══════════════╡
│ 39.1           ┆ 18.7          │
│ 37.8           ┆ 18.3          │
│ 47.5           ┆ 15.0          │
│ 49.9           ┆ 16.1          │
│ 46.5           ┆ 17.9          │
│ 51.3           ┆ 19.2          │
└────────────────┴───────────────┘
```

To keep the page short we'll print `.columns` for the rest — the data comes
along too, of course:

```python
print(penguins.select(col.species, ends_with("_mm")).columns)
print(penguins.select(contains("length")).columns)
print(penguins.select(matches(r"^bill_.*_mm$")).columns)
print(penguins.select(where(is_numeric)).columns)
print(penguins.select(where(is_string)).columns)
print(penguins.select(col.year, everything()).columns)
```

```text
['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
['bill_length_mm', 'flipper_length_mm']
['bill_length_mm', 'bill_depth_mm']
['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'year']
['species', 'island']
['year', 'species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
```

Two ordering rules worth knowing: a selector expands its matches in schema
order, and the first mention of a column fixes its position — that's why
`select(col.year, everything())` puts `year` first without duplicating it.

### Dropping columns with `-`

Negate a column reference or a whole selector with the unary minus. A call
made *only* of negations means "everything except":

```python
print(penguins.select(-col.island).columns)
print(penguins.select(-starts_with("bill"), -col.year).columns)
```

```text
['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'year']
['species', 'island', 'flipper_length_mm']
```

## Reordering with `relocate()`

`select()` reorders but forces you to spell out what to keep. `relocate()`
moves the columns you name and leaves everything else alone. With no anchor
the moved columns go to the front; `before=` / `after=` place them relative
to another column (passing both raises an error):

```python
print(penguins.relocate(col.year).columns)
print(penguins.relocate(starts_with("bill"), after=col.flipper_length_mm).columns)
print(penguins.relocate(col.island, before=col.year).columns)
```

```text
['year', 'species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
['species', 'island', 'flipper_length_mm', 'bill_length_mm', 'bill_depth_mm', 'year']
['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'island', 'year']
```

## `across()`: one recipe, many columns

`across(selector, fns)` goes *inside* `mutate()` or `summarize()` as a
positional argument (everything else in those verbs must be a keyword).
The simplest form pairs a selector with a string shortcut:

```python
print(penguins.summarize(across(where(is_numeric), "mean")))
```

```text
# dpyr frame · source: polars · showing 1 of 1 rows
shape: (1, 4)
┌────────────────┬───────────────┬───────────────────┬────────┐
│ bill_length_mm ┆ bill_depth_mm ┆ flipper_length_mm ┆ year   │
│ ---            ┆ ---           ┆ ---               ┆ ---    │
│ f64            ┆ f64           ┆ f64               ┆ f64    │
╞════════════════╪═══════════════╪═══════════════════╪════════╡
│ 45.35          ┆ 17.533333     ┆ 195.833333        ┆ 2007.5 │
└────────────────┴───────────────┴───────────────────┴────────┘
```

Aggregations skip missing values by default, dplyr-style via `na_rm=True`
(see SEMANTICS S2). The shortcut must be one of a fixed list — anything
else fails immediately, while you're still building the expression:

```python
try:
    across(everything(), "avg")
except Exception as e:
    print(e)
```

```text
across() unknown function shortcut 'avg'; known: first, last, max, mean, median, min, n_unique, std, sum, var
```

For several functions per column, pass a list. Output columns are then named
`{col}_{fn}` by default:

```python
print(penguins.summarize(across(starts_with("bill"), ["mean", "std"])).columns)
```

```text
['bill_length_mm_mean', 'bill_length_mm_std', 'bill_depth_mm_mean', 'bill_depth_mm_std']
```

Anywhere a shortcut fits, a lambda fits too — it receives a typed column
expression. A dict picks your own `{fn}` labels:

```python
print(penguins.summarize(across(
    starts_with("bill"),
    {"avg": "mean", "spread": lambda c: c.max() - c.min()},
)))
```

```text
# dpyr frame · source: polars · showing 1 of 1 rows
shape: (1, 4)
┌─────────────────────┬───────────────────────┬────────────────────┬──────────────────────┐
│ bill_length_mm_mean ┆ bill_length_mm_spread ┆ bill_depth_mm_mean ┆ bill_depth_mm_spread │
│ ---                 ┆ ---                   ┆ ---                ┆ ---                  │
│ f64                 ┆ f64                   ┆ f64                ┆ f64                  │
╞═════════════════════╪═══════════════════════╪════════════════════╪══════════════════════╡
│ 45.35               ┆ 13.5                  ┆ 17.533333          ┆ 4.2                  │
└─────────────────────┴───────────────────────┴────────────────────┴──────────────────────┘
```

`names=` overrides the naming template entirely; `{col}` and `{fn}` are the
available placeholders:

```python
print(penguins.summarize(across(where(is_numeric), "mean", names="mean_of_{col}")).columns)
```

```text
['mean_of_bill_length_mm', 'mean_of_bill_depth_mm', 'mean_of_flipper_length_mm', 'mean_of_year']
```

### `across()` in `mutate()`

With a single function the default template is just `{col}`, so `mutate`
transforms the columns **in place**. Note `year` below: it was Int64, and
`int / int` promotes to float (SEMANTICS S4), so a 2007 becomes 200.7 — set
`names=` if you want to keep the originals:

```python
print(penguins.mutate(across(where(is_numeric), lambda c: c / 10)))
print(penguins.mutate(across(ends_with("_mm"), lambda c: c / 25.4, names="{col}_in")).columns)
```

```text
# dpyr frame · source: polars · showing 6 of 6 rows
shape: (6, 6)
┌───────────┬───────────┬────────────────┬───────────────┬───────────────────┬───────┐
│ species   ┆ island    ┆ bill_length_mm ┆ bill_depth_mm ┆ flipper_length_mm ┆ year  │
│ ---       ┆ ---       ┆ ---            ┆ ---           ┆ ---               ┆ ---   │
│ str       ┆ str       ┆ f64            ┆ f64           ┆ f64               ┆ f64   │
╞═══════════╪═══════════╪════════════════╪═══════════════╪═══════════════════╪═══════╡
│ Adelie    ┆ Torgersen ┆ 3.91           ┆ 1.87          ┆ 18.1              ┆ 200.7 │
│ Adelie    ┆ Dream     ┆ 3.78           ┆ 1.83          ┆ 17.4              ┆ 200.8 │
│ Gentoo    ┆ Biscoe    ┆ 4.75           ┆ 1.5           ┆ 21.7              ┆ 200.7 │
│ Gentoo    ┆ Biscoe    ┆ 4.99           ┆ 1.61          ┆ 21.3              ┆ 200.8 │
│ Chinstrap ┆ Dream     ┆ 4.65           ┆ 1.79          ┆ 19.2              ┆ 200.7 │
│ Chinstrap ┆ Dream     ┆ 5.13           ┆ 1.92          ┆ 19.8              ┆ 200.8 │
└───────────┴───────────┴────────────────┴───────────────┴───────────────────┴───────┘
['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'year', 'bill_length_mm_in', 'bill_depth_mm_in', 'flipper_length_mm_in']
```

### Grouping columns are off-limits

`across()` never touches a grouping column, even when the selector matches
it. Here `year` is numeric, but because it's the grouping key it comes
through untouched instead of being averaged into nonsense. Positional
`across()` results come first, then keyword aggregates like `n=n()`; grouped
results are sorted by key (SEMANTICS S7):

```python
print(penguins.group_by(col.year).summarize(across(where(is_numeric), "mean"), n=n()))
```

```text
# dpyr frame · source: polars · showing 2 of 2 rows
shape: (2, 5)
┌──────┬────────────────┬───────────────┬───────────────────┬─────┐
│ year ┆ bill_length_mm ┆ bill_depth_mm ┆ flipper_length_mm ┆ n   │
│ ---  ┆ ---            ┆ ---           ┆ ---               ┆ --- │
│ i64  ┆ f64            ┆ f64           ┆ f64               ┆ i64 │
╞══════╪════════════════╪═══════════════╪═══════════════════╪═════╡
│ 2007 ┆ 44.366667      ┆ 17.2          ┆ 196.666667        ┆ 3   │
│ 2008 ┆ 46.333333      ┆ 17.866667     ┆ 195.0             ┆ 3   │
└──────┴────────────────┴───────────────┴───────────────────┴─────┘
```

The same protection applies in a grouped `mutate`, where each function runs
as a window per group — handy for centering within groups (the result is
still grouped, as the header shows):

```python
centered = (
    penguins
    .group_by(col.species)
    .mutate(across(starts_with("bill"), lambda c: c - c.mean(), names="{col}_centered"))
    .select(col.species, col.bill_length_mm_centered, col.bill_depth_mm_centered)
)
print(centered)
```

```text
# dpyr frame · groups: species · source: polars · showing 6 of 6 rows
shape: (6, 3)
┌───────────┬─────────────────────────┬────────────────────────┐
│ species   ┆ bill_length_mm_centered ┆ bill_depth_mm_centered │
│ ---       ┆ ---                     ┆ ---                    │
│ str       ┆ f64                     ┆ f64                    │
╞═══════════╪═════════════════════════╪════════════════════════╡
│ Adelie    ┆ 0.65                    ┆ 0.2                    │
│ Adelie    ┆ -0.65                   ┆ -0.2                   │
│ Gentoo    ┆ -1.2                    ┆ -0.55                  │
│ Gentoo    ┆ 1.2                     ┆ 0.55                   │
│ Chinstrap ┆ -2.4                    ┆ -0.65                  │
│ Chinstrap ┆ 2.4                     ┆ 0.65                   │
└───────────┴─────────────────────────┴────────────────────────┘
```

## `rename()`

Keyword form, `new_name=old_column`; the old column can be a `col` reference
or a plain string. Unmentioned columns keep their names and positions:

```python
print(penguins.rename(bill_len=col.bill_length_mm, group="species").columns)
```

```text
['group', 'island', 'bill_len', 'bill_depth_mm', 'flipper_length_mm', 'year']
```

## The same code on duckdb

None of the above is polars-specific. Point a frame at a duckdb table and
the identical chain compiles to SQL instead — note `source: duckdb` in the
header, and the group keys still sorted (S7):

```python
con = duckdb.connect()          # in-memory database
con.register("penguins", penguins.collect())
tbl = read(con, "penguins")

print(tbl.group_by(col.species).summarize(across(starts_with("bill"), "mean"), n=n()))
```

```text
# dpyr frame · source: duckdb · showing 3 of 3 rows
shape: (3, 4)
┌───────────┬────────────────┬───────────────┬─────┐
│ species   ┆ bill_length_mm ┆ bill_depth_mm ┆ n   │
│ ---       ┆ ---            ┆ ---           ┆ --- │
│ str       ┆ f64            ┆ f64           ┆ i64 │
╞═══════════╪════════════════╪═══════════════╪═════╡
│ Adelie    ┆ 38.45          ┆ 18.5          ┆ 2   │
│ Chinstrap ┆ 48.9           ┆ 18.55         ┆ 2   │
│ Gentoo    ┆ 48.7           ┆ 15.55         ┆ 2   │
└───────────┴────────────────┴───────────────┴─────┘
```

Where to next: [Grouped data](grouped-data.md) covers everything else
`group_by` changes, and [Expressions & autocompletion](expressions.md)
explains the typed column expressions that lambdas receive.
