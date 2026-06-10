# Grouped data

In pandas and polars, `groupby`/`group_by` is the entrance to an aggregation:
group, aggregate, back to a flat table. dpyr follows dplyr instead:
`group_by` attaches a *persistent grouping* to the dataframe, and every verb you
call afterwards interprets itself per group — `mutate` becomes a window
computation, `filter` tests rows against group-level values, `slice_head`
takes the first rows *of each group*. The grouping sticks until you
`ungroup()` or `summarize` consumes it. All blocks on this page run top to
bottom as one script.

```python
from dpyr import read, col, n, across, where, is_numeric

sales = read({
    "region": ["north", "north", "north", "south", "south", "south", "east", "east"],
    "rep":    ["ana",   "ana",   "bo",    "cy",    "cy",    "dee",   "ed",   "ed"],
    "amount": [100.0,   200.0,   150.0,   300.0,   250.0,   75.0,    500.0,  500.0],
    "units":  [1,       2,       2,       3,       2,       1,       5,      5],
})

g = sales.group_by(col.region)
print(type(g).__name__, g.groups)
print(repr(g).splitlines()[0])
print(type(g.ungroup()).__name__)
```

```text
GroupedDFrame ('region',)
# dpyr frame · groups: region · source: polars · showing 8 of 8 rows
DFrame
```

`group_by` returns a `GroupedDFrame` — a distinct type, not a flag on the
dataframe. Your IDE's completion surface changes with it, the active keys are
always visible in the repr header and on the `.groups` property (a tuple of
column names), and `ungroup()` hands back a plain `DFrame`. No data moved in
any of this: like every dpyr verb, `group_by` only added a node to the lazy
plan.

## `summarize` peels one grouping level

`summarize` collapses each group to a single row — and then drops **only the
innermost grouping key**, keeping the rest (see SEMANTICS S9). With two keys,
one `summarize` gives you per-(region, rep) rows *still grouped by region*:

```python
by_rep = sales.group_by(col.region, col.rep).summarize(total = col.amount.sum())
print(type(by_rep).__name__, by_rep.groups)
print(by_rep.collect())
```

```text
GroupedDFrame ('region',)
shape: (5, 3)
┌────────┬─────┬────────┐
│ region ┆ rep ┆ total  │
│ ---    ┆ --- ┆ ---    │
│ str    ┆ str ┆ f64    │
╞════════╪═════╪════════╡
│ east   ┆ ed  ┆ 1000.0 │
│ north  ┆ ana ┆ 300.0  │
│ north  ┆ bo  ┆ 150.0  │
│ south  ┆ cy  ┆ 550.0  │
│ south  ┆ dee ┆ 75.0   │
└────────┴─────┴────────┘
```

So layered roll-ups are just chained `summarize` calls. A second one
aggregates the per-rep totals within each region and, having peeled the last
key, returns an ungrouped `DFrame`:

```python
by_region = by_rep.summarize(best = col.total.max(), n_reps = n())
print(type(by_region).__name__)
print(by_region.collect())
```

```text
DFrame
shape: (3, 3)
┌────────┬────────┬────────┐
│ region ┆ best   ┆ n_reps │
│ ---    ┆ ---    ┆ ---    │
│ str    ┆ f64    ┆ i64    │
╞════════╪════════╪════════╡
│ east   ┆ 1000.0 ┆ 1      │
│ north  ┆ 300.0  ┆ 2      │
│ south  ┆ 550.0  ┆ 2      │
└────────┴────────┴────────┘
```

Two pinned details visible above: grouped results come back sorted by the
group keys on both backends (S7), and `n()` always produces Int64 (S13).

## Grouped `mutate` and `filter` are window operations

On a grouped dataframe, aggregates inside `mutate` are evaluated **per group and
broadcast back to every row** — what SQL calls a window function, and what
pandas spells `transform`. The dataframe keeps its original height:

```python
dev = sales.group_by(col.region).mutate(
    region_mean = col.amount.mean(),
    deviation   = col.amount - col.amount.mean(),
)
print(dev.collect())
```

```text
shape: (8, 6)
┌────────┬─────┬────────┬───────┬─────────────┬─────────────┐
│ region ┆ rep ┆ amount ┆ units ┆ region_mean ┆ deviation   │
│ ---    ┆ --- ┆ ---    ┆ ---   ┆ ---         ┆ ---         │
│ str    ┆ str ┆ f64    ┆ i64   ┆ f64         ┆ f64         │
╞════════╪═════╪════════╪═══════╪═════════════╪═════════════╡
│ north  ┆ ana ┆ 100.0  ┆ 1     ┆ 150.0       ┆ -50.0       │
│ north  ┆ ana ┆ 200.0  ┆ 2     ┆ 150.0       ┆ 50.0        │
│ north  ┆ bo  ┆ 150.0  ┆ 2     ┆ 150.0       ┆ 0.0         │
│ south  ┆ cy  ┆ 300.0  ┆ 3     ┆ 208.333333  ┆ 91.666667   │
│ south  ┆ cy  ┆ 250.0  ┆ 2     ┆ 208.333333  ┆ 41.666667   │
│ south  ┆ dee ┆ 75.0   ┆ 1     ┆ 208.333333  ┆ -133.333333 │
│ east   ┆ ed  ┆ 500.0  ┆ 5     ┆ 500.0       ┆ 0.0         │
│ east   ┆ ed  ┆ 500.0  ┆ 5     ┆ 500.0       ┆ 0.0         │
└────────┴─────┴────────┴───────┴─────────────┴─────────────┘
```

Same rule for `filter`: predicates can compare each row against a per-group
aggregate. East's tie produces two rows — this is a row predicate, not a
"take 1":

```python
best_rows = sales.group_by(col.region).filter(col.amount == col.amount.max())
print(best_rows.collect())
```

```text
shape: (4, 4)
┌────────┬─────┬────────┬───────┐
│ region ┆ rep ┆ amount ┆ units │
│ ---    ┆ --- ┆ ---    ┆ ---   │
│ str    ┆ str ┆ f64    ┆ i64   │
╞════════╪═════╪════════╪═══════╡
│ north  ┆ ana ┆ 200.0  ┆ 2     │
│ south  ┆ cy  ┆ 300.0  ┆ 3     │
│ east   ┆ ed  ┆ 500.0  ┆ 5     │
│ east   ┆ ed  ┆ 500.0  ┆ 5     │
└────────┴─────┴────────┴───────┘
```

Both stay grouped, so you can keep chaining grouped verbs (or `ungroup()`).

## The other verbs respect groups too

**`count(...)`** on a grouped dataframe adds its columns to the existing keys,
counts, and — via the peeling rule — comes back grouped by the *original*
keys, exactly as in dplyr:

```python
per_rep = sales.group_by(col.region).count(col.rep)
print(type(per_rep).__name__, per_rep.groups)
print(per_rep.collect())
```

```text
GroupedDFrame ('region',)
shape: (5, 3)
┌────────┬─────┬─────┐
│ region ┆ rep ┆ n   │
│ ---    ┆ --- ┆ --- │
│ str    ┆ str ┆ i64 │
╞════════╪═════╪═════╡
│ east   ┆ ed  ┆ 2   │
│ north  ┆ ana ┆ 2   │
│ north  ┆ bo  ┆ 1   │
│ south  ┆ cy  ┆ 2   │
│ south  ┆ dee ┆ 1   │
└────────┴─────┴─────┘
```

**`slice_head(n)`** takes the first `n` rows of *each* group, with the
original column order restored even where the engine would move keys first
(S25). **`slice_min` / `slice_max`** rank within each group and keep ties by
default (`with_ties=False` for exactly `n` rows); their output is arranged by
the slicing key, which is why east's tied pair lands at the bottom below:

```python
print(sales.group_by(col.region).slice_head(1).collect())
print(sales.group_by(col.region).slice_min(col.amount, n = 1).collect())
```

```text
shape: (3, 4)
┌────────┬─────┬────────┬───────┐
│ region ┆ rep ┆ amount ┆ units │
│ ---    ┆ --- ┆ ---    ┆ ---   │
│ str    ┆ str ┆ f64    ┆ i64   │
╞════════╪═════╪════════╪═══════╡
│ north  ┆ ana ┆ 100.0  ┆ 1     │
│ south  ┆ cy  ┆ 300.0  ┆ 3     │
│ east   ┆ ed  ┆ 500.0  ┆ 5     │
└────────┴─────┴────────┴───────┘
shape: (4, 4)
┌────────┬─────┬────────┬───────┐
│ region ┆ rep ┆ amount ┆ units │
│ ---    ┆ --- ┆ ---    ┆ ---   │
│ str    ┆ str ┆ f64    ┆ i64   │
╞════════╪═════╪════════╪═══════╡
│ south  ┆ dee ┆ 75.0   ┆ 1     │
│ north  ┆ ana ┆ 100.0  ┆ 1     │
│ east   ┆ ed  ┆ 500.0  ┆ 5     │
│ east   ┆ ed  ┆ 500.0  ┆ 5     │
└────────┴─────┴────────┴───────┘
```

**`distinct(...)`** automatically adds the group keys to the deduplication
set: `distinct(col.rep)` below really means "distinct (region, rep) pairs",
and `region` stays in the output — 5 unique pairs, not 5 unique reps:

```python
pairs = sales.group_by(col.region).distinct(col.rep)
print(pairs.columns, len(pairs))
```

```text
['region', 'rep'] 5
```

**`select(...)`** refuses to drop a group key — a dataframe can't end up grouped
by a column it no longer has. And **`across(...)`** goes the other way: its
selector never matches the keys, so `where(is_numeric)` means "every numeric
column *except* `region`" and you can't aggregate your own keys by accident:

```python
print(sales.group_by(col.region).select(col.amount).columns)
print(sales.group_by(col.region).summarize(across(where(is_numeric), "mean")).collect())
```

```text
['region', 'amount']
shape: (3, 3)
┌────────┬────────────┬──────────┐
│ region ┆ amount     ┆ units    │
│ ---    ┆ ---        ┆ ---      │
│ str    ┆ f64        ┆ f64      │
╞════════╪════════════╪══════════╡
│ east   ┆ 500.0      ┆ 5.0      │
│ north  ┆ 150.0      ┆ 1.666667 │
│ south  ┆ 208.333333 ┆ 2.0      │
└────────┴────────────┴──────────┘
```

## `persist()` keeps the grouping

`persist()` materializes the plan so far and rebinds the dataframe to the result
(on duckdb, a temp table). It's a checkpoint, not a semantic boundary: the
grouping is reattached to the snapshot, mirroring dplyr's `compute()`.

```python
checkpoint = sales.group_by(col.region).mutate(
    deviation = col.amount - col.amount.mean(),
).persist()
print(type(checkpoint).__name__, checkpoint.groups)
```

```text
GroupedDFrame ('region',)
```

## The classic gotcha: forgetting `ungroup()`

Because grouping is sticky, an aggregate that you *meant* to run over the
whole dataframe quietly runs per group instead. The same `mutate` produces shares
of the **region** total or of the **grand** total depending on whether the
grouping is still active:

```python
grouped_share = g.mutate(share = col.amount / col.amount.sum()).pull(col.share)
global_share  = g.ungroup().mutate(share = col.amount / col.amount.sum()).pull(col.share)
print([round(s, 3) for s in grouped_share])   # sums to 1.0 per region
print([round(s, 3) for s in global_share])    # sums to 1.0 overall
```

```text
[0.222, 0.444, 0.333, 0.48, 0.4, 0.12, 0.5, 0.5]
[0.048, 0.096, 0.072, 0.145, 0.12, 0.036, 0.241, 0.241]
```

The same trap exists for `slice_head(5)` (5 rows *per group*, not 5 rows
total). dpyr softens it where it can: the repr header always shows
`groups: ...`, `GroupedDFrame` is a separate type so it shows up in type
hints and tracebacks, and ambiguous operations fail loudly — passing a
still-grouped dataframe as the right-hand side of a join raises
`GroupError: joining a grouped dataframe is not supported; ungroup() it first`.

## Same semantics on duckdb

When the source is a duckdb table, the same chains compile to SQL window
functions and aggregates, and the grouped behaviors are identical — enforced
by backend-agreement fuzzing. One caveat: row order after `distinct` is
unspecified on both backends, so pin it with `arrange()` (S21).

```python
import duckdb

con = duckdb.connect()                                    # in-memory
sales_pl = sales.collect()                                # polars DataFrame
con.execute("CREATE TABLE sales AS SELECT * FROM sales_pl")
tbl = read(con, "sales")
print(
    tbl.group_by(col.region)
       .filter(col.amount == col.amount.max())
       .ungroup()
       .distinct()
       .arrange(col.region)
       .collect()
)
```

```text
shape: (3, 4)
┌────────┬─────┬────────┬───────┐
│ region ┆ rep ┆ amount ┆ units │
│ ---    ┆ --- ┆ ---    ┆ ---   │
│ str    ┆ str ┆ f64    ┆ i64   │
╞════════╪═════╪════════╪═══════╡
│ east   ┆ ed  ┆ 500.0  ┆ 5     │
│ north  ┆ ana ┆ 200.0  ┆ 2     │
│ south  ┆ cy  ┆ 300.0  ┆ 3     │
└────────┴─────┴────────┴───────┘
```
