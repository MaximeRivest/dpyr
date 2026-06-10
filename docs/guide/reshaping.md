# Reshaping

dpyr ships the tidyr-flavored reshaping verbs as plain `DFrame` methods:
`pivot_longer`, `pivot_wider`, `separate`, `unite`, and `relocate`. They run
on either backend — every example below works the same whether your frame
came from `read()` on plain Python data (polars) or on a duckdb connection
(SQL pushdown). If you know
`pl.DataFrame.unpivot`/`pivot` or pandas `melt`/`pivot_table`, the mapping is
direct; the differences are in the details (null handling, duplicate keys),
and those are pinned down in [SEMANTICS.md](../SEMANTICS.md).

All code on this page runs top to bottom as one script.

## Wide to long: `pivot_longer`

Pass the columns to stack as a list. Their *names* land in the `names_to`
column, their *values* in the `values_to` column; everything you didn't list
(here `country`) is repeated as an identifier.

```python
from dpyr import read, col, starts_with, INT64

wide = read({
    "country":  ["Canada", "France", "Japan"],
    "pop_2000": [30.7, 60.9, 126.8],
    "pop_2010": [34.0, 65.0, 128.1],
    "pop_2020": [38.0, 67.4, None],   # no 2020 figure for Japan
})

long = wide.pivot_longer(
    [col.pop_2000, col.pop_2010, col.pop_2020],
    names_to="year",
    values_to="pop",
)
print(long)
```

```text
# dpyr frame · source: polars · showing 9 of 9 rows
shape: (9, 3)
┌─────────┬──────────┬───────┐
│ country ┆ year     ┆ pop   │
│ ---     ┆ ---      ┆ ---   │
│ str     ┆ str      ┆ f64   │
╞═════════╪══════════╪═══════╡
│ Canada  ┆ pop_2000 ┆ 30.7  │
│ France  ┆ pop_2000 ┆ 60.9  │
│ Japan   ┆ pop_2000 ┆ 126.8 │
│ Canada  ┆ pop_2010 ┆ 34.0  │
│ France  ┆ pop_2010 ┆ 65.0  │
│ Japan   ┆ pop_2010 ┆ 128.1 │
│ Canada  ┆ pop_2020 ┆ 38.0  │
│ France  ┆ pop_2020 ┆ 67.4  │
│ Japan   ┆ pop_2020 ┆ null  │
└─────────┴──────────┴───────┘
```

Three things to notice:

- **Selectors work in the list.** `[starts_with("pop_")]` picks the same
  three columns without spelling them out — any tidyselect helper
  (`ends_with`, `contains`, `matches`, `where(...)`) is accepted.
- **Nulls are kept.** Japan's missing 2020 value becomes a regular `null`
  row, not a dropped one. There is no `values_drop_na` switch; filter
  explicitly when you want the dense version.
- **Row order is unspecified** after a pivot (see SEMANTICS S21). Polars
  happens to stack column-by-column here, but duckdb may differ — finish
  with `arrange()` whenever order matters.

```python
long = wide.pivot_longer([starts_with("pop_")], names_to="year", values_to="pop")
print(long.filter(~col.pop.is_na()).count())   # drop the null row by hand
```

```text
shape: (1, 1)
┌─────┐
│ n   │
│ --- │
│ i64 │
╞═════╡
│ 8   │
└─────┘
```

The `names_to` column is always a string column built from the source column
names. Turning `"pop_2020"` into a usable integer year is an ordinary
`mutate`:

```python
tidy = (
    long
    .mutate(year=col.year.str_replace("pop_", "").cast(INT64))
    .arrange(col.country, col.year)
)
print(tidy.slice_head(4))
```

```text
shape: (4, 3)
┌─────────┬──────┬──────┐
│ country ┆ year ┆ pop  │
│ ---     ┆ ---  ┆ ---  │
│ str     ┆ i64  ┆ f64  │
╞═════════╪══════╪══════╡
│ Canada  ┆ 2000 ┆ 30.7 │
│ Canada  ┆ 2010 ┆ 34.0 │
│ Canada  ┆ 2020 ┆ 38.0 │
│ France  ┆ 2000 ┆ 60.9 │
└─────────┴──────┴──────┘
```

## Long to wide: `pivot_wider`

`pivot_wider` is the inverse: one new column per distinct value of
`names_from`, filled from `values_from`. Every column you don't name acts as
an identifier. The round trip recovers the original table exactly:

```python
wide_again = long.pivot_wider(names_from=col.year, values_from=col.pop)
print(wide_again)
print(wide_again.collect().equals(wide.collect()))
```

```text
shape: (3, 4)
┌─────────┬──────────┬──────────┬──────────┐
│ country ┆ pop_2000 ┆ pop_2010 ┆ pop_2020 │
│ ---     ┆ ---      ┆ ---      ┆ ---      │
│ str     ┆ f64      ┆ f64      ┆ f64      │
╞═════════╪══════════╪══════════╪══════════╡
│ Canada  ┆ 30.7     ┆ 34.0     ┆ 38.0     │
│ France  ┆ 60.9     ┆ 65.0     ┆ 67.4     │
│ Japan   ┆ 126.8    ┆ 128.1    ┆ null     │
└─────────┴──────────┴──────────┴──────────┘
True
```

One quirk worth knowing: because the output schema depends on the *data*
(the distinct values of `names_from`), `pivot_wider` materializes its input
immediately rather than staying lazy. That keeps `df.c.<TAB>` completion and
schema errors working on the result.

### Duplicate keys warn and keep the first value

If an (id, name) combination maps to more than one value, dplyr would nest
them into list-columns. dpyr instead emits a `UserWarning` and keeps the
first value per key (SEMANTICS S26):

```python
dup = read({
    "id":  [1, 1, 2],
    "key": ["a", "a", "a"],
    "val": [10, 99, 20],   # id=1, key="a" appears twice
})
dwide = dup.pivot_wider(names_from=col.key, values_from=col.val)
print(dwide)
```

```text
UserWarning: pivot_wider(): values are not uniquely identified for 1 key
combination(s); keeping the first value (dplyr would build list-columns)

shape: (2, 2)
┌─────┬─────┐
│ id  ┆ a   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 1   ┆ 10  │
│ 2   ┆ 20  │
└─────┴─────┘
```

The warning fires at the `pivot_wider()` call itself (the eager
materialization above), so it points at the offending line, not at a distant
`collect()`. Related S26 detail: a `null` in the `names_from` column becomes
a column literally named `"null"`.

### No id columns

With nothing left over to identify rows, you simply get a one-row frame:

```python
totals = read({"metric": ["rows", "cols", "cells"], "value": [120, 8, 960]})
print(totals.pivot_wider(names_from=col.metric, values_from=col.value))
```

```text
shape: (1, 3)
┌──────┬──────┬───────┐
│ rows ┆ cols ┆ cells │
│ ---  ┆ ---  ┆ ---   │
╞══════╪══════╪═══════╡
│ 120  ┆ 8    ┆ 960   │
└──────┴──────┴───────┘
```

## Splitting a column: `separate`

`separate` splits one string column into several. Unlike tidyr — where `sep`
is a regex — dpyr's `sep` is a **literal string**, default `"_"` (SEMANTICS
S31). Pieces are assigned left-to-right, so too few pieces leave the
trailing columns `null`; extra pieces beyond `into` are silently dropped.

```python
codes = read({
    "code":  ["FR-75-Paris", "CA-QC", None, "JP-13-Tokyo-extra"],
    "score": [1, 2, 3, 4],
})
print(codes.separate(col.code, into=["country", "region", "city"], sep="-"))
```

```text
shape: (4, 4)
┌─────────┬────────┬───────┬───────┐
│ country ┆ region ┆ city  ┆ score │
│ ---     ┆ ---    ┆ ---   ┆ ---   │
│ str     ┆ str    ┆ str   ┆ i64   │
╞═════════╪════════╪═══════╪═══════╡
│ FR      ┆ 75     ┆ Paris ┆ 1     │
│ CA      ┆ QC     ┆ null  ┆ 2     │
│ null    ┆ null   ┆ null  ┆ 3     │
│ JP      ┆ 13     ┆ Tokyo ┆ 4     │
└─────────┴────────┴───────┴───────┘
```

The new columns take the source column's position. A `null` input yields
`null` in every piece, and `"-extra"` from the last row is gone. Pass
`remove=False` to also keep the original, placed right after the pieces:

```python
kept = codes.separate(col.code, into=["country", "region", "city"],
                      sep="-", remove=False)
print(kept.collect().columns)
```

```text
['country', 'region', 'city', 'code', 'score']
```

## Gluing columns: `unite`

`unite` is the inverse: paste several columns into one string column, joined
by `sep` (default `"_"`), inserted where the first source column was.
Missing values render as the string `"NA"` by default — matching dplyr, and
surprising if you expected null propagation (SEMANTICS S32):

```python
parts = read({
    "id":    [1, 2, 3, 4],
    "year":  ["2024", "2025", None, None],
    "month": ["01", None, "07", None],
})
print(parts.unite("ym", [col.year, col.month], sep="-"))
```

```text
shape: (4, 2)
┌─────┬─────────┐
│ id  ┆ ym      │
│ --- ┆ ---     │
│ i64 ┆ str     │
╞═════╪═════════╡
│ 1   ┆ 2024-01 │
│ 2   ┆ 2025-NA │
│ 3   ┆ NA-07   │
│ 4   ┆ NA-NA   │
└─────┴─────────┘
```

`na_rm=True` drops missing pieces instead; a row where *every* piece is
missing collapses to the empty string `""`, not null:

```python
print(parts.unite("ym", [col.year, col.month], sep="-", na_rm=True))
print(parts.unite("ym", [col.year, col.month], sep="-", remove=False).collect().columns)
```

```text
shape: (4, 2)
┌─────┬─────────┐
│ id  ┆ ym      │
╞═════╪═════════╡
│ 1   ┆ 2024-01 │
│ 2   ┆ 2025    │
│ 3   ┆ 07      │
│ 4   ┆         │
└─────┴─────────┘
['id', 'ym', 'year', 'month']
```

As with `separate`, `remove=False` keeps the source columns (right after the
new one).

## Reordering columns: `relocate`

Reshapes often leave columns in an awkward order. `relocate` moves columns
without touching data: by default to the front, or anchored with `before=` /
`after=`. Selectors work here too.

```python
print(tidy.relocate(col.pop).collect().columns)                 # to the front
print(tidy.relocate(col.year, after=col.pop).collect().columns) # anchored
```

```text
['pop', 'country', 'year']
['country', 'pop', 'year']
```

## Same verbs on duckdb

Nothing changes on the SQL backend — `pivot_longer` compiles to pushed-down
SQL and stays lazy; `pivot_wider` materializes (as above) to learn its
schema:

```python
import duckdb

con = duckdb.connect()   # in-memory database
con.execute("""
    CREATE TABLE sales AS
    SELECT * FROM (VALUES ('north', 10, 12), ('south', 7, 9)) t(region, q1, q2)
""")

sales = read(con, "sales")
sales_long = sales.pivot_longer([col.q1, col.q2], names_to="quarter", values_to="units")
print(sales_long.arrange(col.region, col.quarter))
print(sales_long.pivot_wider(names_from=col.quarter, values_from=col.units))
```

```text
# dpyr frame · source: duckdb · showing 4 of 4 rows
shape: (4, 3)
┌────────┬─────────┬───────┐
│ region ┆ quarter ┆ units │
╞════════╪═════════╪═══════╡
│ north  ┆ q1      ┆ 10    │
│ north  ┆ q2      ┆ 12    │
│ south  ┆ q1      ┆ 7     │
│ south  ┆ q2      ┆ 9     │
└────────┴─────────┴───────┘
# dpyr frame · source: duckdb · showing 2 of 2 rows
shape: (2, 3)
┌────────┬─────┬─────┐
│ region ┆ q1  ┆ q2  │
╞════════╪═════╪═════╡
│ north  ┆ 10  ┆ 12  │
│ south  ┆ 7   ┆ 9   │
└────────┴─────┴─────┘
```
