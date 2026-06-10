# Window functions

A regular `mutate()` expression like `col.mass * 2` only needs the current row.
A **window function** produces one value per row too, but it computes that value
by peeking at *other* rows: the previous row (`lag`), the row's position in a
sorted order (`min_rank`), or everything up to here (`cum_sum`). If you know
`Series.shift`/`rank`/`cum_sum` from pandas or polars, these are the same ideas
— with one twist that dpyr inherits from dplyr: windows automatically respect
the dataframe's active `group_by()` and any `arrange()` that came before them.

All of the functions below are top-level imports and are used inside
`mutate()` or `filter()` — never `summarize()` (more on that at the end).

## Offsets: `lag` and `lead`

`lag(expr)` pulls the value from the previous row; `lead(expr)` from the next.
Rows that fall off the edge get null, which makes day-over-day changes a
one-liner.

```python
from dpyr import read, col, lag, lead, desc

prices = read({
    "day":   [1, 2, 3, 4, 5],
    "close": [100, 120, 120, 90, 150],
})

prices.mutate(
    prev = lag(col.close),
    nxt  = lead(col.close),
    change = col.close - lag(col.close),
)
```

```text
┌─────┬───────┬──────┬──────┬────────┐
│ day ┆ close ┆ prev ┆ nxt  ┆ change │
╞═════╪═══════╪══════╪══════╪════════╡
│ 1   ┆ 100   ┆ null ┆ 120  ┆ null   │
│ 2   ┆ 120   ┆ 100  ┆ 120  ┆ 20     │
│ 3   ┆ 120   ┆ 120  ┆ 90   ┆ 0      │
│ 4   ┆ 90    ┆ 120  ┆ 150  ┆ -30    │
│ 5   ┆ 150   ┆ 90   ┆ null ┆ 60     │
└─────┴───────┴──────┴──────┴────────┘
```

Both take `n` (how far to shift, default 1) and `default` (what to use instead
of null at the edges):

```python
prices.mutate(
    two_back = lag(col.close, n=2),
    padded   = lag(col.close, default=0),
)
```

```text
┌─────┬───────┬──────────┬────────┐
│ day ┆ close ┆ two_back ┆ padded │
╞═════╪═══════╪══════════╪════════╡
│ 1   ┆ 100   ┆ null     ┆ 0      │
│ 2   ┆ 120   ┆ null     ┆ 100    │
│ 3   ┆ 120   ┆ 100      ┆ 120    │
│ 4   ┆ 90    ┆ 120      ┆ 120    │
│ 5   ┆ 150   ┆ 120      ┆ 90     │
└─────┴───────┴──────────┴────────┘
```

## Ranks

Four ranking functions, differing only in how they treat ties:

- `row_number()` — 1, 2, 3, … in current row order; never ties, takes no argument.
- `min_rank(e)` — ties share the *lowest* rank and leave a gap after
  (Olympic medals: two silvers, no bronze). This is SQL's `RANK()`.
- `dense_rank(e)` — ties share a rank but no gap follows.
- `percent_rank(e)` — `min_rank` rescaled to `[0, 1]`. dpyr follows dplyr's
  formula `(min_rank - 1) / (number of non-missing values - 1)` on both
  backends, rather than raw SQL `PERCENT_RANK()` which counts NULL rows
  (see SEMANTICS S30).

Nulls never receive a rank — they rank as null:

```python
from dpyr import row_number, min_rank, dense_rank, percent_rank

scores = read({
    "player": ["ana", "bo", "cy", "dee", "eli"],
    "score":  [10, 20, 20, 30, None],
})

scores.mutate(
    rn = row_number(),
    mr = min_rank(col.score),
    dr = dense_rank(col.score),
    pr = percent_rank(col.score),
)
```

```text
┌────────┬───────┬─────┬──────┬──────┬──────────┐
│ player ┆ score ┆ rn  ┆ mr   ┆ dr   ┆ pr       │
╞════════╪═══════╪═════╪══════╪══════╪══════════╡
│ ana    ┆ 10    ┆ 1   ┆ 1    ┆ 1    ┆ 0.0      │
│ bo     ┆ 20    ┆ 2   ┆ 2    ┆ 2    ┆ 0.333333 │
│ cy     ┆ 20    ┆ 3   ┆ 2    ┆ 2    ┆ 0.333333 │
│ dee    ┆ 30    ┆ 4   ┆ 4    ┆ 3    ┆ 1.0      │
│ eli    ┆ null  ┆ 5   ┆ null ┆ null ┆ null     │
└────────┴───────┴─────┴──────┴──────┴──────────┘
```

Note the `min_rank` gap: two players at rank 2, then dee jumps to 4, while
`dense_rank` gives her 3.

Ranks go smallest-first. To rank the biggest value as #1, wrap the expression
in `desc()`:

```python
scores.mutate(place = min_rank(desc(col.score)))
```

```text
┌────────┬───────┬───────┐
│ player ┆ score ┆ place │
╞════════╪═══════╪═══════╡
│ ana    ┆ 10    ┆ 4     │
│ bo     ┆ 20    ┆ 2     │
│ cy     ┆ 20    ┆ 2     │
│ dee    ┆ 30    ┆ 1     │
│ eli    ┆ null  ┆ null  │
└────────┴───────┴───────┘
```

## Running aggregates: `cum_sum`, `cum_min`, `cum_max`

These accumulate from the first row down. The null rule matters and is a
deliberate divergence from R (SEMANTICS S29): a null input produces a null
*at that row only*, and the accumulation keeps going underneath. (R's
`cumsum` would poison every subsequent row; SQL window sums would skip the
null entirely. dpyr pins the polars behavior on both backends.)

```python
from dpyr import cum_sum, cum_min, cum_max

ledger = read({
    "month": [1, 2, 3, 4, 5],
    "delta": [5, None, -2, 8, None],
})

ledger.mutate(
    total = cum_sum(col.delta),
    low   = cum_min(col.delta),
    high  = cum_max(col.delta),
)
```

```text
┌───────┬───────┬───────┬──────┬──────┐
│ month ┆ delta ┆ total ┆ low  ┆ high │
╞═══════╪═══════╪═══════╪══════╪══════╡
│ 1     ┆ 5     ┆ 5     ┆ 5    ┆ 5    │
│ 2     ┆ null  ┆ null  ┆ null ┆ null │
│ 3     ┆ -2    ┆ 3     ┆ -2   ┆ 5    │
│ 4     ┆ 8     ┆ 11    ┆ -2   ┆ 8    │
│ 5     ┆ null  ┆ null  ┆ null ┆ null │
└───────┴───────┴───────┴──────┴──────┘
```

Month 3 reads `3 = 5 + (-2)`: the running total survived the null at month 2.

## Windows follow your groups — and your sort order

On a grouped dataframe, every window restarts per group. No `over(...)` clause to
write; the dataframe's grouping *is* the window partition:

```python
sales = read({
    "store": ["n", "n", "n", "s", "s", "s"],
    "qtr":   [2, 1, 3, 1, 3, 2],
    "rev":   [110, 100, 130, 80, 95, 90],
})

sales.group_by(col.store).mutate(prev = lag(col.rev))
```

```text
┌───────┬─────┬─────┬──────┐
│ store ┆ qtr ┆ rev ┆ prev │
╞═══════╪═════╪═════╪══════╡
│ n     ┆ 2   ┆ 110 ┆ null │
│ n     ┆ 1   ┆ 100 ┆ 110  │
│ n     ┆ 3   ┆ 130 ┆ 100  │
│ s     ┆ 1   ┆ 80  ┆ null │
│ s     ┆ 3   ┆ 95  ┆ 80   │
│ s     ┆ 2   ┆ 90  ┆ 95   │
└───────┴─────┴─────┴──────┘
```

Each store's first row gets null — but the rows are still in their original,
unsorted order, so "previous" means "previous as the data happens to sit".
For time-based logic, `arrange()` first; offsets and cumulatives then operate
in that order (SEMANTICS S28):

```python
(sales
    .arrange(col.qtr)
    .group_by(col.store)
    .mutate(prev = lag(col.rev), running = cum_sum(col.rev))
)
```

```text
┌───────┬─────┬─────┬──────┬─────────┐
│ store ┆ qtr ┆ rev ┆ prev ┆ running │
╞═══════╪═════╪═════╪══════╪═════════╡
│ n     ┆ 1   ┆ 100 ┆ null ┆ 100     │
│ s     ┆ 1   ┆ 80  ┆ null ┆ 80      │
│ n     ┆ 2   ┆ 110 ┆ 100  ┆ 210     │
│ s     ┆ 2   ┆ 90  ┆ 80   ┆ 170     │
│ n     ┆ 3   ┆ 130 ┆ 110  ┆ 340     │
│ s     ┆ 3   ┆ 95  ┆ 90   ┆ 265     │
└───────┴─────┴─────┴──────┴─────────┘
```

Now `prev` is genuinely "last quarter, same store". The exact same chain runs
on duckdb — dpyr compiles the pending sort into ordered window dataframes:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE TABLE sales (store TEXT, qtr BIGINT, rev BIGINT)")
con.execute(
    "INSERT INTO sales VALUES "
    "('n', 2, 110), ('n', 1, 100), ('n', 3, 130), "
    "('s', 1, 80), ('s', 3, 95), ('s', 2, 90)"
)

(read(con, "sales")
    .arrange(col.qtr)
    .group_by(col.store)
    .mutate(prev = lag(col.rev), running = cum_sum(col.rev))
)
# -> identical rows, header says `source: duckdb`
```

## Windows inside `filter()`: top-n per group

Because grouped `filter()` evaluates its predicate per group, ranking inside a
filter is the classic "best n rows per group" recipe:

```python
(sales
    .group_by(col.store)
    .filter(min_rank(desc(col.rev)) <= 2)
    .ungroup()
    .arrange(col.store, desc(col.rev))
)
```

```text
┌───────┬─────┬─────┐
│ store ┆ qtr ┆ rev │
╞═══════╪═════╪═════╡
│ n     ┆ 3   ┆ 130 │
│ n     ┆ 2   ┆ 110 │
│ s     ┆ 3   ┆ 95  │
│ s     ┆ 2   ┆ 90  │
└───────┴─────┴─────┘
```

## The shortcut: `slice_min` / `slice_max`

That pattern is common enough to have verbs.
`slice_max(col.rev, n=2)` on a grouped dataframe keeps each store's two highest
revenues — it is implemented as exactly the `min_rank` filter above:

```python
sales.group_by(col.store).slice_max(col.rev, n=2)
# -> same four rows as the filter version (still grouped by store)
```

Because it ranks with `min_rank`, **ties are kept by default**, so you can get
more than `n` rows back — the dplyr behavior. Pass `with_ties=False` to get an
exact count (it switches to sort-then-take-n):

```python
scores.slice_max(col.score, n=2)
```

```text
┌────────┬───────┐
│ player ┆ score │
╞════════╪═══════╡
│ dee    ┆ 30    │
│ bo     ┆ 20    │
│ cy     ┆ 20    │   <- 3 rows for n=2: bo and cy tie at 20
└────────┴───────┘
```

```python
scores.slice_max(col.score, n=2, with_ties=False)   # exactly 2 rows: dee, bo
```

`slice_min` is the same thing ranked upward.

## `summarize()` refuses windows

A window returns one value per row; `summarize()` must collapse each group to
one row. Handing it a window is a shape error, and dpyr rejects it at
plan-build time — before any data moves — with a pointer to the right verb:

```python
try:
    sales.group_by(col.store).summarize(prev = lag(col.rev))
except Exception as err:
    print(f"{type(err).__name__}: {err}")
```

```text
ExprTypeError: summarize(prev=...) must aggregate (use .mean(), n(), ...) (window functions like lag/min_rank belong in mutate()): lag(col.rev, 1, default=lit(None))
```

If you want "last value per group", that's an ordered aggregate
(`col.rev.last()` after an `arrange()`), not a window — see SEMANTICS S28.
