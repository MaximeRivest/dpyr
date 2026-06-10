# Joins

dpyr gives you dplyr's two join families as methods on any frame:

- **Mutating joins** — `inner_join`, `left_join`, `right_join`, `full_join` —
  pull columns from a second table into the first.
- **Filtering joins** — `semi_join`, `anti_join` — use the second table only
  to decide which rows of the first to keep. No columns are added, and rows
  are never duplicated.

Everything below runs identically on the polars and duckdb backends. We'll
use two small tables: one row per customer, and one row per order (a
customer can place several orders, and an order can reference a customer we
don't know about).

```python
from dpyr import from_dict, col

customers = from_dict({
    "cust_id": [1, 2, 3, 4],
    "name":    ["Ada", "Grace", "Linus", "Guido"],
    "city":    ["London", "Arlington", "Helsinki", "Haarlem"],
})

orders = from_dict({
    "cust_id": [1, 1, 3, 5],
    "item":    ["keyboard", "monitor", "laptop", "mouse"],
    "total":   [55.0, 210.0, 1450.0, 25.0],
})
```

## Mutating joins

Keys are passed as `on=col.name` (one key) or `on=[col.a, col.b]` (several).
The key column appears once in the result, and left columns keep their
original positions (see SEMANTICS S22).

`inner_join` keeps only rows whose key exists on both sides. Customer 1
matched two orders, so Ada appears twice:

```python
customers.inner_join(orders, on=col.cust_id)
```

```text
# dpyr frame · source: polars · showing 3 of 3 rows
shape: (3, 5)
┌─────────┬───────┬──────────┬──────────┬────────┐
│ cust_id ┆ name  ┆ city     ┆ item     ┆ total  │
╞═════════╪═══════╪══════════╪══════════╪════════╡
│ 1       ┆ Ada   ┆ London   ┆ keyboard ┆ 55.0   │
│ 1       ┆ Ada   ┆ London   ┆ monitor  ┆ 210.0  │
│ 3       ┆ Linus ┆ Helsinki ┆ laptop   ┆ 1450.0 │
└─────────┴───────┴──────────┴──────────┴────────┘
```

`left_join` keeps every left row; right columns fill with `null` where
nothing matched. This is the workhorse — "decorate my table with whatever
extra info exists":

```python
customers.left_join(orders, on=col.cust_id)
```

```text
shape: (5, 5)
┌─────────┬───────┬───────────┬──────────┬────────┐
│ cust_id ┆ name  ┆ city      ┆ item     ┆ total  │
╞═════════╪═══════╪═══════════╪══════════╪════════╡
│ 1       ┆ Ada   ┆ London    ┆ keyboard ┆ 55.0   │
│ 1       ┆ Ada   ┆ London    ┆ monitor  ┆ 210.0  │
│ 2       ┆ Grace ┆ Arlington ┆ null     ┆ null   │
│ 3       ┆ Linus ┆ Helsinki  ┆ laptop   ┆ 1450.0 │
│ 4       ┆ Guido ┆ Haarlem   ┆ null     ┆ null   │
└─────────┴───────┴───────────┴──────────┴────────┘
```

`right_join` is the mirror image (every order survives, including the one
from unknown customer 5), and `full_join` keeps everything from both sides:

```python
customers.right_join(orders, on=col.cust_id)   # 4 rows: cust_id 5 has null name/city
customers.full_join(orders, on=col.cust_id)    # 6 rows: union of both sides
```

Running that full join, we got cust_ids in the order `1, 1, 3, 5, 2, 4` —
matched pairs first, unmatched leftovers appended. That ordering is *not* a
guarantee; see [below](#row-order-is-unspecified-pin-it-with-arrange).

### Multiple keys

Pass a list to match on a composite key:

```python
sales = from_dict({
    "year":    [2024, 2024, 2025],
    "region":  ["east", "west", "east"],
    "revenue": [120.0, 95.0, 140.0],
})
targets = from_dict({
    "year":   [2024, 2024, 2025, 2025],
    "region": ["east", "west", "east", "west"],
    "target": [100.0, 100.0, 130.0, 110.0],
})

sales.inner_join(targets, on=[col.year, col.region])
# 3 rows × 4 cols (year, region, revenue, target);
# the 2025/west target had no matching sales row, so it's dropped
```

## Overlapping column names get `.x` / `.y` suffixes

When both tables have a non-key column with the same name, dpyr keeps both
and disambiguates with dplyr's suffixes: `.x` for the left table, `.y` for
the right (SEMANTICS S11) — not pandas' `_x`/`_y` or polars' `_right`:

```python
jan = from_dict({"sku": ["A1", "B2"], "price": [9.99, 24.00]})
feb = from_dict({"sku": ["A1", "B2"], "price": [10.49, 22.50]})

jan.inner_join(feb, on=col.sku)
```

```text
shape: (2, 3)
┌─────┬─────────┬─────────┐
│ sku ┆ price.x ┆ price.y │
╞═════╪═════════╪═════════╡
│ A1  ┆ 9.99    ┆ 10.49   │
│ B2  ┆ 24.0    ┆ 22.5    │
└─────┴─────────┴─────────┘
```

Dots in column names are awkward to type with `col.`, so for downstream work
you may prefer your own suffixes:

```python
jan.inner_join(feb, on=col.sku, suffix=("_jan", "_feb"))
# same two rows, columns now: sku, price_jan, price_feb
```

## Missing values in join keys: `na_matches`

SQL says `NULL = NULL` is unknown, so SQL joins silently drop rows whose key
is missing. dplyr instead treats two missing keys as equal — and dpyr
follows dplyr by default on **both** backends (SEMANTICS S10):

```python
left  = from_dict({"k": ["x", None], "lv": [1, 2]})
right = from_dict({"k": ["x", None], "rv": [10, 20]})

left.inner_join(right, on=col.k)              # default: na_matches="na"
```

```text
shape: (2, 3)
┌──────┬─────┬─────┐
│ k    ┆ lv  ┆ rv  │
╞══════╪═════╪═════╡
│ x    ┆ 1   ┆ 10  │
│ null ┆ 2   ┆ 20  │
└──────┴─────┴─────┘
```

Pass `na_matches="never"` to get the SQL behavior — the `null` keys no
longer pair up:

```python
left.inner_join(right, on=col.k, na_matches="never")
# 1 row: only k="x" survives
```

On duckdb, the default compiles to `IS NOT DISTINCT FROM` so both modes are
available there too. All four mutating joins accept `na_matches` (and
`suffix`); the filtering joins below take only `on=`.

## Filtering joins: `semi_join` and `anti_join`

A `semi_join` keeps the left rows that have at least one match; an
`anti_join` keeps the ones that have none. Neither adds columns:

```python
customers.semi_join(orders, on=col.cust_id)
# 2 rows, same 3 columns as customers: Ada and Linus (they ordered)

customers.anti_join(orders, on=col.cust_id)
# 2 rows: Grace and Guido (they never ordered)
```

The key difference from `inner_join` + `select`: a semi join never
multiplies rows. Customer 1 has two orders but still appears once. Prefer
`semi_join` whenever the question is "which rows of A are in B?" rather
than "what does B say about A?".

## Duplicate keys multiply rows

Mutating joins emit one output row per *pair* of matching rows. If a key is
duplicated on both sides, you get a per-key cross product — 2 × 2 = 4 rows
here:

```python
runs   = from_dict({"day": ["mon", "mon"], "run":   ["r1", "r2"]})
alerts = from_dict({"day": ["mon", "mon"], "alert": ["disk", "cpu"]})

runs.inner_join(alerts, on=col.day)
```

```text
shape: (4, 3)
┌─────┬─────┬───────┐
│ day ┆ run ┆ alert │
╞═════╪═════╪═══════╡
│ mon ┆ r1  ┆ disk  │
│ mon ┆ r2  ┆ disk  │
│ mon ┆ r1  ┆ cpu   │
│ mon ┆ r2  ┆ cpu   │
└─────┴─────┴───────┘
```

This is correct join algebra, but when it's a surprise it usually means the
right table wasn't unique per key. Compare `tbl.distinct(col.k).count()`
against `tbl.count()` to check, or reach for `semi_join` if you only wanted
existence.

## Row order is unspecified — pin it with `arrange()`

dpyr deliberately does **not** promise any row order after a join
(SEMANTICS S21). dplyr preserves left order, but hash joins in polars and
duckdb don't, and forcing it would cost a sort on every join. The
`1, 1, 3, 5, 2, 4` order of the earlier full join is just what the engine
emitted that time. If order matters, say so explicitly:

```python
from dpyr import desc

(
    customers
    .full_join(orders, on=col.cust_id)
    .arrange(col.cust_id, desc(col.total))
)
```

```text
shape: (6, 5)
┌─────────┬───────┬───────────┬──────────┬────────┐
│ cust_id ┆ name  ┆ city      ┆ item     ┆ total  │
╞═════════╪═══════╪═══════════╪══════════╪════════╡
│ 1       ┆ Ada   ┆ London    ┆ monitor  ┆ 210.0  │
│ 1       ┆ Ada   ┆ London    ┆ keyboard ┆ 55.0   │
│ 2       ┆ Grace ┆ Arlington ┆ null     ┆ null   │
│ 3       ┆ Linus ┆ Helsinki  ┆ laptop   ┆ 1450.0 │
│ 4       ┆ Guido ┆ Haarlem   ┆ null     ┆ null   │
│ 5       ┆ null  ┆ null      ┆ mouse    ┆ 25.0   │
└─────────┴───────┴───────────┴──────────┴────────┘
```

## Joins on duckdb: one connection per plan

Joining two duckdb-backed frames pushes the whole thing down as a single SQL
query — as long as both frames come from the **same** connection:

```python
import duckdb
from dpyr import from_duckdb

con = duckdb.connect()
con.execute("CREATE TABLE people AS SELECT * FROM (VALUES (1, 'Ada'), (2, 'Grace')) t(pid, name)")
con.execute("CREATE TABLE badges AS SELECT * FROM (VALUES (1, 'gold'), (2, 'silver'), (2, 'bronze')) t(pid, badge)")

people = from_duckdb(con, "people")
badges = from_duckdb(con, "badges")

people.inner_join(badges, on=col.pid).arrange(col.pid, col.badge)
```

```text
# dpyr frame · source: duckdb · showing 3 of 3 rows
shape: (3, 3)
┌─────┬───────┬────────┐
│ pid ┆ name  ┆ badge  │
╞═════╪═══════╪════════╡
│ 1   ┆ Ada   ┆ gold   │
│ 2   ┆ Grace ┆ bronze │
│ 2   ┆ Grace ┆ silver │
└─────┴───────┴────────┘
```

Frames from *different* connections can't be compiled into one query.
Building the join succeeds (it's just a plan), but materializing raises a
`BackendError` (SEMANTICS S27):

```python
other_con = duckdb.connect()
other_con.execute("CREATE TABLE badges2 AS SELECT * FROM (VALUES (1, 'gold')) t(pid, badge)")
stray = from_duckdb(other_con, "badges2")

try:
    people.inner_join(stray, on=col.pid).collect()
except Exception as e:
    print(f"{type(e).__name__}: {e}")
```

```text
BackendError: plan joins tables from different duckdb connections; persist one side or use a single connection
```

The simplest fix is to move one side's data onto the shared connection —
for example, round-trip it through arrow:

```python
con.register("badges_local", stray.to_polars().to_arrow())
people.inner_join(from_duckdb(con, "badges_local"), on=col.pid)
# 1 row: pid=1, Ada, gold — and the whole join runs inside con
```

The same rule applies one level up: a plan can't mix a duckdb frame with a
polars frame. That also fails at collect time, with
`BackendError: plan mixes polars and duckdb sources; collect one side first
(e.g. .persist() or .to_polars()) before joining across backends`.

## Cheat sheet

| Verb | Rows kept | Columns added | Can duplicate rows |
|---|---|---|---|
| `inner_join` | matches only | yes | yes |
| `left_join` | all left | yes (null-filled) | yes |
| `right_join` | all right | yes (null-filled) | yes |
| `full_join` | all of both | yes (null-filled) | yes |
| `semi_join` | left rows with a match | no | no |
| `anti_join` | left rows without a match | no | no |

All four mutating joins accept `suffix=(".x", ".y")` and
`na_matches="na" | "never"`; the filtering joins take only `on=`.
