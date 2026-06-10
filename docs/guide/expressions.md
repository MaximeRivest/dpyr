# Expressions and autocompletion

Everything you pass to a verb — `col.kg > 5`, `col.yield_kg.mean()`,
`if_else(...)` — is an **expression**: a small immutable tree describing a
computation. Nothing runs when you build one. The dataframe validates it against
its schema the moment a verb receives it (so mistakes surface instantly), and
the work happens later, on polars or duckdb, when the dataframe materializes.
This page tours the expression toolkit and the three tiers of autocompletion
built on it: `col`, `df.c`, and `dpyr stubgen`.

## `col` describes, the dataframe executes

`col.<name>` is a free-floating column reference — dpyr's counterpart of
`pl.col("name")` — whose operators and methods keep growing the tree. Print
one and you see the IR, not data:

```python
from dpyr import col

bmi = (col.mass / (col.height / 100) ** 2).round(1)
print(repr(bmi))
```

```text
round((col.mass / pow((col.height / lit(100)), lit(2))), lit(1))
```

That repr is canonical: dpyr hashes it to fingerprint plans and cache
results. Plain Python values (`100`, `"kale"`, `date(...)`) become literals
automatically, and the same `bmi` object works on any dataframe with `mass` and
`height` columns, on either backend.

## `&`, `|`, `~` — not `and`, `or`, `not`

Python's keyword operators force their operands through `bool()`, and a
description of a computation has no truth value. dpyr makes the failure loud
and names the fix:

```python
try:
    col.kg > 5 and col.organic
except TypeError as e:
    print(f"{type(e).__name__}: {e}")

both   = (col.kg > 5) & col.organic    # AND
either = (col.kg > 5) | ~col.organic   # OR, NOT
```

```text
ExprTypeError: a dpyr expression is not a Python boolean. Use & | ~ instead of and/or/not, and is_in() instead of `in`.
```

Parenthesize comparisons next to `&`/`|` — the bitwise operators bind
tighter, so `col.kg > 5 & col.organic` parses as `col.kg > (5 & col.organic)`.
And use `.is_in([...])` where you'd reach for Python's `in`, which also
routes through `bool()` and hits the same error.

## A dataframe to play with

```python
from datetime import date
from dpyr import read

plants = read({
    "plant":    ["Roma tomato", "Cherry tomato", "Basil", "Squash", "Pepper"],
    "family":   ["nightshade", "nightshade", "herb", "cucurbit", "nightshade"],
    "sown":     [date(2026, 3, 14), date(2026, 3, 20), date(2026, 4, 2),
                 date(2026, 5, 1), date(2026, 3, 18)],
    "rows":     [4, 6, 2, 3, 5],
    "yield_kg": [41.5, 33.2, None, float("nan"), 12.9],
})
print(plants)
```

```text
# dpyr dataframe · source: polars · showing 5 of 5 rows
shape: (5, 5)
┌───────────────┬────────────┬────────────┬──────┬──────────┐
│ plant         ┆ family     ┆ sown       ┆ rows ┆ yield_kg │
│ ---           ┆ ---        ┆ ---        ┆ ---  ┆ ---      │
│ str           ┆ str        ┆ date       ┆ i64  ┆ f64      │
╞═══════════════╪════════════╪════════════╪══════╪══════════╡
│ Roma tomato   ┆ nightshade ┆ 2026-03-14 ┆ 4    ┆ 41.5     │
│ Cherry tomato ┆ nightshade ┆ 2026-03-20 ┆ 6    ┆ 33.2     │
│ Basil         ┆ herb       ┆ 2026-04-02 ┆ 2    ┆ null     │
│ Squash        ┆ cucurbit   ┆ 2026-05-01 ┆ 3    ┆ NaN      │
│ Pepper        ┆ nightshade ┆ 2026-03-18 ┆ 5    ┆ 12.9     │
└───────────────┴────────────┴────────────┴──────┴──────────┘
```

`yield_kg` carries both a `null` (a missing value) and a `NaN` (a real float,
the not-a-number value). dpyr keeps the two distinct (SEMANTICS S1) — they
behave differently below.

## Methods follow the column's type

| Works on | Per-row | Aggregating |
|---|---|---|
| numeric | `.abs()` `.round(digits)` `.floor()` `.ceiling()` `.log()` `.exp()` `.sqrt()` | `.mean()` `.median()` `.sum()` `.std()` `.var()` |
| string | `.str_detect(pat)` `.str_replace(pat, repl)` `.str_to_lower()` `.str_to_upper()` `.str_len()` | |
| date / datetime | `.year()` `.month()` `.day()` | |
| any | `.is_na()` `.is_in(values)` `.between(lo, hi)` `.cast(dtype)` | `.min()` `.max()` `.first()` `.last()` `.n_unique()` |

Aggregates skip missing values by default; pass `na_rm=False` to propagate
them instead (SEMANTICS S2). String patterns are regular expressions on both
backends, and `str_replace` rewrites the first match (stringr-style, unlike
Python's replace-all `str.replace`). Temporal accessors return integers. A
quick pass over each family — note how the `null` and `NaN` rows flow through
arithmetic untouched:

```python
print(plants.mutate(per_row = (col.yield_kg / col.rows).round(2)).pull(col.per_row))
print(plants.filter(col.plant.str_detect("tomato")).pull(col.plant))
print(plants.mutate(p = col.plant.str_replace(" tomato", "")).pull(col.p))
print(plants.filter(col.sown.month() == 3).pull(col.plant))
```

```text
[10.38, 5.53, None, nan, 2.58]
['Roma tomato', 'Cherry tomato']
['Roma', 'Cherry', 'Basil', 'Squash', 'Pepper']
['Roma tomato', 'Cherry tomato', 'Pepper']
```

(`.pull(col.x)` collects one column as a Python list — handy for compact
output here.)

## Missing values, membership, ranges, casts

`.is_na()` uses R's definition of "missing": true for `null` **and** for
`NaN` on float columns (SEMANTICS S1), on both backends — so it catches both
oddball rows:

```python
print(plants.filter(col.yield_kg.is_na()).pull(col.plant))
```

```text
['Basil', 'Squash']
```

Outside of `.is_na()`, though, `NaN` is an ordinary float that **compares
greater than every number** on both engines, while comparisons against `null`
yield `null`, which `filter` drops (SEMANTICS S12). A threshold filter
therefore keeps the NaN row and silently sheds the null one:

```python
print(plants.filter(col.yield_kg > 30).pull(col.plant))
```

```text
['Roma tomato', 'Cherry tomato', 'Squash']
```

If your floats may contain NaN, add `~col.yield_kg.is_na()` before
thresholding. The remaining utilities:

```python
from dpyr import FLOAT64

print(plants.filter(col.family.is_in(["herb", "cucurbit"])).pull(col.plant))
print(plants.filter(col.rows.between(3, 5)).pull(col.plant))   # inclusive ends
print(plants.mutate(rows = col.rows.cast(FLOAT64)).schema["rows"])
```

```text
['Basil', 'Squash']
['Roma tomato', 'Squash', 'Pepper']
Float64
```

`.is_in()` on a missing value returns null rather than R's `FALSE`
(SEMANTICS S24), and `.cast()` takes the dtype constants dpyr exports:
`INT64`, `FLOAT64`, `BOOL`, `STR`, `DATE`, `DATETIME`.

## Conditionals: `if_else`, `case_when`, `coalesce`, `replace_na`

`case_when` takes `(condition, value)` pairs, first match wins, and
`default=` covers the rest (no match and no default gives a missing value,
SEMANTICS S15). Branch dtypes must unify — mixing strings and ints across
branches is a build-time `ExprTypeError`, not a runtime surprise.

```python
from dpyr import if_else, case_when, coalesce, replace_na

graded = plants.mutate(
    scale  = if_else(col.rows >= 5, "big", "small"),
    grade  = case_when(
        (col.yield_kg >= 30, "great"),
        (col.yield_kg >= 10, "fine"),
        default = "unweighed",
    ),
    filled = replace_na(col.yield_kg, 0.0),
    capped = coalesce(col.yield_kg, col.rows * 5.0),  # estimate when missing
)
print(graded.select(col.plant, col.scale, col.grade, col.filled, col.capped))
```

```text
# dpyr dataframe · source: polars · showing 5 of 5 rows
shape: (5, 5)
┌───────────────┬───────┬───────────┬────────┬────────┐
│ plant         ┆ scale ┆ grade     ┆ filled ┆ capped │
│ ---           ┆ ---   ┆ ---       ┆ ---    ┆ ---    │
│ str           ┆ str   ┆ str       ┆ f64    ┆ f64    │
╞═══════════════╪═══════╪═══════════╪════════╪════════╡
│ Roma tomato   ┆ small ┆ great     ┆ 41.5   ┆ 41.5   │
│ Cherry tomato ┆ big   ┆ great     ┆ 33.2   ┆ 33.2   │
│ Basil         ┆ small ┆ unweighed ┆ 0.0    ┆ 10.0   │
│ Squash        ┆ small ┆ great     ┆ NaN    ┆ NaN    │
│ Pepper        ┆ big   ┆ fine      ┆ 12.9   ┆ 12.9   │
└───────────────┴───────┴───────────┴────────┴────────┘
```

Look at the Squash row: it graded "great" (NaN ≥ 30 is true, as above), and
neither `replace_na` nor `coalesce` touched it — both fill *nulls* only,
while NaN is a value. To treat NaN as missing in a fill, route through
`.is_na()`:

```python
print(plants.mutate(
    y0 = if_else(col.yield_kg.is_na(), 0.0, col.yield_kg),
).pull(col.y0))
```

```text
[41.5, 33.2, 0.0, 0.0, 12.9]
```

## The same expressions on duckdb

Expressions are backend-agnostic; the duckdb compiler turns the identical
tree into SQL (`case_when` → `CASE WHEN`, `.is_na()` →
`IS NULL OR isnan(...)`):

```python
import duckdb

con = duckdb.connect()   # in-memory
con.execute("""
    CREATE TABLE sales AS SELECT * FROM (VALUES
        ('Roma tomato', 41.5), ('Basil', NULL), ('Pepper', 12.9)
    ) AS t(plant, yield_kg)
""")
sales = read(con, "sales")
print(sales.mutate(
    grade = case_when(
        (col.yield_kg >= 30, "great"),
        (col.yield_kg >= 10, "fine"),
        default = "unweighed",
    ),
    missing = col.yield_kg.is_na(),
))
```

```text
# dpyr dataframe · source: duckdb · showing 3 of 3 rows
shape: (3, 4)
┌─────────────┬──────────┬───────────┬─────────┐
│ plant       ┆ yield_kg ┆ grade     ┆ missing │
│ ---         ┆ ---      ┆ ---       ┆ ---     │
│ str         ┆ f64      ┆ str       ┆ bool    │
╞═════════════╪══════════╪═══════════╪═════════╡
│ Roma tomato ┆ 41.5     ┆ great     ┆ false   │
│ Basil       ┆ null     ┆ unweighed ┆ true    │
│ Pepper      ┆ 12.9     ┆ fine      ┆ false   │
└─────────────┴──────────┴───────────┴─────────┘
```

## Mistakes surface on your line

Verbs validate every expression against the schema before returning — pure
metadata work, so it's instant. A wrong column name raises with a
did-you-mean suggestion, and dpyr strips its internals from the traceback:
exactly two stack frames, your call plus one re-raise inside dpyr (paths below come
from running this guide as a script):

```python
import traceback

try:
    plants.filter(col.yeild_kg > 10)
except Exception:
    traceback.print_exc(chain=False)
```

```text
Traceback (most recent call last):
  File "/tmp/expr_full2.py", line 17, in <module>
    plants.filter(col.yeild_kg > 10)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/home/maxime/Projects/r_ports_to_py/dpyr/src/dpyr/frame.py", line 51, in verb
    raise err.with_traceback(None) from None
dpyr.errors.ColumnNotFoundError: column 'yeild_kg' not found in filter(). Did you mean 'yield_kg'? Available columns: plant, family, sown, rows, yield_kg
```

Type mistakes get the same treatment: comparing a string to an int, summing a
date, or mixing incompatible `case_when` branches all raise `ExprTypeError`
on the offending verb call.

## `df.c`: the schema-bound, type-aware proxy

`col` accepts any name and any method, deferring all checks to the verb. The
dataframe-bound proxy `df.c` knows the live schema: in Jupyter or any REPL,
`plants.c.<TAB>` completes real column names, and what comes back is a
*typed* expression class:

```python
print(type(plants.c.yield_kg).__name__, "/", type(plants.c.plant).__name__,
      "/", type(plants.c.sown).__name__)
```

```text
NumExpr / StrExpr / TemporalExpr
```

`NumExpr` has no `.str_detect`, `StrExpr` has no `.mean` — completion menus
only offer methods that make sense, and a wrong-type method fails immediately
at expression-build time, before any verb or backend is involved. Typos fail
at attribute access with the same did-you-mean:

```python
try:
    plants.c.plant.mean()
except TypeError as e:
    print(f"{type(e).__name__}: {e}")

try:
    plants.c.famly
except KeyError as e:
    print(f"{type(e).__name__}: {e}")
```

```text
ExprTypeError: .mean() is not available on a StrExpr
ColumnNotFoundError: column 'famly' not found in df.c. Did you mean 'family'? Available columns: plant, family, sown, rows, yield_kg
```

### Lambda verbs

`filter`, `mutate`, and `summarize` accept callables and pass them `df.c`, so
you get the typed proxy without naming the dataframe twice — handy mid-chain,
where the intermediate dataframe has no variable name:

```python
print(plants.filter(lambda c: c.rows >= 5).pull(col.plant))
print(plants.mutate(per_row = lambda c: (c.yield_kg / c.rows).round(1))
      .slice_head(2).pull(col.per_row))
```

```text
['Cherry tomato', 'Pepper']
[10.4, 5.5]
```

## Static completion anywhere: `dpyr stubgen`

Runtime completion needs a live kernel. For static completion and
type-checking in any editor, the `dpyr stubgen` CLI reads parquet/csv schemas
and writes a typed module: one `ColsProxy` subclass per file with a typed
attribute per column, plus a loader returning `DFrame[YourCols]`. Shell usage
is `dpyr stubgen data/*.parquet -o schemas.py`; here it is end-to-end (temp
paths will differ on your machine):

```python
import subprocess, sys, tempfile
from pathlib import Path

workdir = Path(tempfile.mkdtemp())
plants.write(workdir / "plants.parquet")

subprocess.run(["dpyr", "stubgen", str(workdir / "plants.parquet"),
                "-o", str(workdir / "garden_schemas.py")], check=True)
print((workdir / "garden_schemas.py").read_text())
```

```text
"""Generated by `dpyr stubgen` — do not edit by hand."""

from typing import cast

from dpyr import DFrame, read
from dpyr.expr import BoolExpr, NumExpr, StrExpr, TemporalExpr
from dpyr.frame import ColsProxy

class PlantsCols(ColsProxy):
    plant: StrExpr
    family: StrExpr
    sown: TemporalExpr
    rows: NumExpr
    yield_kg: NumExpr


def load_plants() -> DFrame[PlantsCols]:
    return cast(DFrame[PlantsCols], read('/tmp/tmpvbdbwibt/plants.parquet'))


plants: DFrame[PlantsCols] = load_plants()
```

Import from that module and the `DFrame[PlantsCols]` annotation flows through
the chain: pyright/mypy infer `c: PlantsCols` inside lambda verbs, so
`c.yie<TAB>` completes and `c.plant.mean()` is flagged *in the editor*,
before anything runs — and it works at runtime too:

```python
sys.path.insert(0, str(workdir))
from garden_schemas import plants as typed_plants

print(typed_plants.filter(lambda c: c.rows >= 5).pull("plant"))
```

```text
['Cherry tomato', 'Pepper']
```

dpyr ships a `py.typed` marker, so type checkers pick up its inline
annotations with zero configuration — generated schema modules, lambda verbs,
and ordinary chains all type-check out of the box.

## Where next

- [Grouped data](grouped-data.md) — aggregates and windows per group
- [Window functions](window-functions.md) — `lag`, ranks, cumulative sums
- [Backends](backends.md) — polars vs duckdb, caching, `persist()`
