# Backends: polars and duckdb

Every dpyr chain builds the same logical plan + expression IR
([DESIGN §4](../DESIGN.md)); the backend is decided by where the data came
from. A polars-backed source compiles to a `LazyFrame`, a duckdb-backed
source compiles to SQL — and the *same chain returns the same rows on both*.
That promise is enforced by a Hypothesis fuzzer that runs random verb chains
on both engines and compares bit-for-bit, plus differential tests against
goldens produced by real dplyr (the oracle). You pick a backend for data
location and performance, never for behavior.

## The polars backend

Anything that starts from in-process data or local files runs on polars:
`from_dict`, `from_polars`, `from_pandas`, `read_parquet`, `read_csv`.

```python
import polars as pl
import dpyr
from dpyr import from_dict, from_polars, col, n, desc

sales = from_dict({
    "city":  ["Hull", "Hull", "Aylmer", "Aylmer", "Wakefield", "Wakefield"],
    "month": [1, 2, 1, 2, 1, 2],
    "units": [40, 35, 21, 28, 12, None],
    "price": [2.5, 2.5, 3.0, 3.0, 4.0, 4.0],
})
print(sales)
```

```text
# dpyr frame · source: polars · showing 6 of 6 rows
shape: (6, 4)
┌───────────┬───────┬───────┬───────┐
│ city      ┆ month ┆ units ┆ price │
│ ---       ┆ ---   ┆ ---   ┆ ---   │
│ str       ┆ i64   ┆ i64   ┆ f64   │
╞═══════════╪═══════╪═══════╪═══════╡
│ Hull      ┆ 1     ┆ 40    ┆ 2.5   │
│ Hull      ┆ 2     ┆ 35    ┆ 2.5   │
│ Aylmer    ┆ 1     ┆ 21    ┆ 3.0   │
│ Aylmer    ┆ 2     ┆ 28    ┆ 3.0   │
│ Wakefield ┆ 1     ┆ 12    ┆ 4.0   │
│ Wakefield ┆ 2     ┆ null  ┆ 4.0   │
└───────────┴───────┴───────┴───────┘
```

`read_parquet` and `read_csv` are scans, not loads — the file is only read
when something materializes, and only the columns/rows the plan needs:

```python
import tempfile, pathlib

tmp = pathlib.Path(tempfile.mkdtemp())
sales.collect().write_parquet(tmp / "sales.parquet")

from dpyr import read_parquet
sales_pq = read_parquet(str(tmp / "sales.parquet"))
print(sales_pq.schema)
```

```text
{'city': Str, 'month': Int64, 'units': Int64, 'price': Float64}
```

A whole chain compiles to **one** polars `LazyFrame`, so polars' optimizer
sees everything at once — filters get pushed into the parquet scan, unused
columns are never decoded, and no intermediate frame is allocated between
verbs:

```python
revenue = (
    sales_pq
    .filter(~col.units.is_na())
    .mutate(revenue = col.units * col.price)
    .group_by(col.city)
    .summarize(total = col.revenue.sum())
    .arrange(desc(col.total))
)
print(revenue)
```

```text
# dpyr frame · source: polars · showing 3 of 3 rows
shape: (3, 2)
┌───────────┬───────┐
│ city      ┆ total │
│ ---       ┆ ---   │
│ str       ┆ f64   │
╞═══════════╪═══════╡
│ Hull      ┆ 187.5 │
│ Aylmer    ┆ 147.0 │
│ Wakefield ┆ 48.0  │
└───────────┴───────┘
```

## The duckdb backend

For data that already lives in a database — or is too big to pull into
memory — point dpyr at a duckdb connection. `duckdb.connect()` gives an
in-memory database; `duckdb.connect("warehouse.db")` opens a file.
`from_duckdb(con, table)` wraps an existing table, `read_sql(con, query)`
wraps an arbitrary query as a source:

```python
import duckdb
from dpyr import from_duckdb, read_sql

con = duckdb.connect()   # in-memory; pass a path for a persistent file
con.execute("""
    CREATE TABLE deliveries (city VARCHAR, month INTEGER, km DOUBLE);
    INSERT INTO deliveries VALUES
        ('Hull', 1, 12.0),      ('Hull', 2, 15.5),
        ('Aylmer', 1, 30.2),    ('Aylmer', 2, 28.9),
        ('Wakefield', 1, 55.0), ('Wakefield', 2, 51.3);
""")
deliveries = from_duckdb(con, "deliveries")

by_city = (
    deliveries
    .group_by(col.city)
    .summarize(trips = n(), total_km = col.km.sum())
    .arrange(desc(col.total_km))
)
print(by_city)

feb = read_sql(con, "SELECT city, km FROM deliveries WHERE month = 2")
print(feb.schema)
```

```text
# dpyr frame · source: duckdb · showing 3 of 3 rows
shape: (3, 3)
┌───────────┬───────┬──────────┐
│ city      ┆ trips ┆ total_km │
│ ---       ┆ ---   ┆ ---      │
│ str       ┆ i64   ┆ f64      │
╞═══════════╪═══════╪══════════╡
│ Wakefield ┆ 2     ┆ 106.3    │
│ Aylmer    ┆ 2     ┆ 59.1     │
│ Hull      ┆ 2     ┆ 27.5     │
└───────────┴───────┴──────────┘
{'city': Str, 'km': Float64}
```

The verbs compile to a single SQL statement that duckdb executes entirely on
its side; only the result crosses back into Python. The generated SQL is an
implementation detail with no public API — it carries semantics shims
(NULLS-LAST stable sorts, BIGINT counts, floor-mod: SEMANTICS S3/S13/S23)
that make it noisier than hand-written SQL. Peeking via the *internal*
compiler entry point, `by_city` would send:

```python
from dpyr.duckdb_backend import final_sql   # private, may change
print(final_sql(by_city.plan))
```

```text
SELECT *, row_number() OVER () AS "__rn1" FROM (SELECT "city", CAST(count(*) AS BIGINT) AS "trips", COALESCE(sum("km"), 0.0) AS "total_km" FROM (SELECT "city", "month", "km" FROM "deliveries") t GROUP BY "city") t ORDER BY "total_km" DESC NULLS LAST, "__rn1" ASC
```

One duckdb-side effect *is* observable: `.persist()` checkpoints a frame as a
`TEMP TABLE` on your connection (polars frames persist as in-memory frames
instead). Temp tables vanish when the connection closes:

```python
snap = by_city.persist()
print(con.execute(
    "SELECT table_name, temporary FROM duckdb_tables() ORDER BY table_name"
).fetchall())
```

```text
[('deliveries', False), ('dpyr_persist_2', True)]
```

## Both backends hand you polars

`collect()` returns a `polars.DataFrame` regardless of engine — polars is the
interchange format. `to_pandas()` converts from there (it needs pandas
installed: `pip install 'dpyr[pandas]'`):

```python
print(type(revenue.collect()), type(by_city.collect()))
print(type(by_city.to_pandas()))
```

```text
<class 'polars.dataframe.frame.DataFrame'> <class 'polars.dataframe.frame.DataFrame'>
<class 'pandas.DataFrame'>
```

## The result cache (and when to clear it)

Results are cached keyed by plan hash, so re-displaying a frame in a notebook
costs nothing. The flip side: a duckdb table mutated *outside* dpyr has the
same plan hash, so cached results go stale. `dpyr.cache_clear()` drops the
cache; `.persist()` is the explicit "snapshot now" alternative:

```python
print(len(deliveries))                                      # collects, caches
con.execute("INSERT INTO deliveries VALUES ('Chelsea', 2, 18.0)")
print(len(deliveries))                                      # cache hit: stale!
dpyr.cache_clear()
print(len(deliveries))                                      # recomputed
```

```text
6
6
7
```

File-backed sources are *not* exempt. `read_parquet`/`read_csv` tag the
source with the file's path + mtime + size — but only once, at construction
(`_file_token` in `frame.py`). A frame you are already holding keeps that
original tag, so editing the file on disk does not change its plan hash:
it keeps returning the cached rows. `cache_clear()` doesn't rescue it
either — the captured scan pinned the old file's metadata, and collecting
now raises a polars `ComputeError`. To see the new contents, construct a
*new* source with `read_parquet(path)`: the fresh mtime/size give it a
fresh plan hash that never collides with the stale entry. (This is why
re-running the notebook cell that creates the source picks up file edits —
that's reconstruction, not cache invalidation.)

```python
pq = str(tmp / "sales.parquet")
held = read_parquet(pq)
print(len(held))                            # collects and caches
sales.collect().head(2).write_parquet(pq)   # rewrite the file: 2 rows now
print(len(held))                            # same tag, same hash: stale
print(len(read_parquet(pq)))                # new source, new tag: fresh

dpyr.cache_clear()
try:
    len(held)                               # the held frame is unrecoverable
except pl.exceptions.ComputeError:
    print("held frame: scan pinned the old file's metadata")
```

```text
6
6
2
held frame: scan pinned the old file's metadata
```

## Opting out of eager display

By default a frame's repr collects a preview (display-eager, DESIGN §3).
`.lazy()` returns a frame that never executes implicitly — only `.collect()`
(or another explicit export) runs it. `.eager()` flips it back, and
`dpyr.options.interactive = False` turns implicit execution off globally for
production pipelines:

```python
prod = by_city.lazy()
print(prod)            # schema only, nothing executed
fresh = prod.collect() # explicit is fine (now includes Chelsea: 4 rows)

dpyr.options.interactive = False   # same effect, process-wide
print(by_city)
dpyr.options.interactive = True
```

```text
# dpyr frame · source: duckdb (lazy)
# columns: city <Str>, trips <Int64>, total_km <Float64>
# dpyr frame · source: duckdb (lazy)
# columns: city <Str>, trips <Int64>, total_km <Float64>
```

## One exception to pushdown: `pivot_wider`

`pivot_wider`'s output columns come from data values, so its schema can't be
known from metadata alone. On *either* backend, dpyr implicitly persists the
input, then pivots with polars. It works the same on a duckdb frame — just
know that this step runs in-process, not in the database:

```python
wide = deliveries.pivot_wider(names_from=col.month, values_from=col.km)
print(wide)
```

```text
# dpyr frame · source: duckdb · showing 4 of 4 rows
shape: (4, 3)
┌───────────┬──────┬──────┐
│ city      ┆ 1    ┆ 2    │
│ ---       ┆ ---  ┆ ---  │
│ str       ┆ f64  ┆ f64  │
╞═══════════╪══════╪══════╡
│ Hull      ┆ 12.0 ┆ 15.5 │
│ Aylmer    ┆ 30.2 ┆ 28.9 │
│ Wakefield ┆ 55.0 ┆ 51.3 │
│ Chelsea   ┆ null ┆ 18.0 │
└───────────┴──────┴──────┘
```

Duplicate keys warn and keep the first value (see SEMANTICS S26).

## One plan, one engine

A single plan cannot span engines, and dpyr refuses to copy data behind your
back. Joining a polars frame to a duckdb frame fails at collect time with
instructions:

```python
try:
    sales.inner_join(deliveries, on=[col.city, col.month]).collect()
except dpyr.DpyrError as e:
    print(e)
```

```text
plan mixes polars and duckdb sources; collect one side first (e.g. .persist() or .to_polars()) before joining across backends
```

The fix is explicit: `from_polars(duck_frame.collect())` to pull the duckdb
side over, or load the polars side into duckdb yourself. Two tables from
*different* duckdb connections are rejected too (SEMANTICS S27):

```python
con2 = duckdb.connect()
con2.execute(
    "CREATE TABLE depots AS SELECT * FROM (VALUES ('Hull', 4), ('Aylmer', 2)) t(city, docks)"
)
try:
    deliveries.inner_join(from_duckdb(con2, "depots"), on=col.city).collect()
except dpyr.DpyrError as e:
    print(e)
```

```text
plan joins tables from different duckdb connections; persist one side or use a single connection
```

## One dtype system

dpyr pins six data dtypes (plus `Null` for all-missing literals), and both
backends are normalized to them on ingest:

| dpyr dtype | from polars | from duckdb |
|---|---|---|
| `Int64` | all int widths cast up (`Int8`…`UInt64`) | `TINYINT`…`HUGEINT`, unsigned ints |
| `Float64` | `Float32` cast up | `FLOAT`, `DOUBLE`, `DECIMAL(…)` |
| `Bool` | `Boolean` | `BOOLEAN` |
| `Str` | `String` | `VARCHAR` |
| `Date` | `Date` | `DATE` |
| `Datetime` | cast to microsecond precision | `TIMESTAMP` |

```python
odd = from_polars(pl.DataFrame({
    "tiny": pl.Series([1, 2], dtype=pl.Int8),
    "f32":  pl.Series([1.5, 2.5], dtype=pl.Float32),
}))
print(odd.schema)
print(deliveries.schema)   # INTEGER and DOUBLE arrived as Int64/Float64 too
```

```text
{'tiny': Int64, 'f32': Float64}
{'city': Str, 'month': Int64, 'km': Float64}
```

Columns outside this set are rejected up front rather than half-supported —
`from_duckdb` on a table with a `BLOB` or nested column raises
`DpyrError: column 'payload' has unsupported duckdb type BLOB`. Int64 is the
one integer type (S5), counts are Int64 (S13), `int / int` gives Float64
(S4), and datetimes are microsecond-precision (S16).

## Which backend?

- **polars** — data already in memory, parquet/CSV files that fit on one
  machine, tight notebook loops. Lowest overhead per query.
- **duckdb** — data already in a `.db` file or larger than RAM, sources you
  want to define in SQL (`read_sql`), or pipelines where the heavy lifting
  should stay inside the database and only summaries come back.

Since both return polars frames and obey the same semantics, switching is a
one-line change at the source constructor — the chain below it stays
identical. The few places where engine behavior genuinely differs (and what
dpyr does about each) are cataloged row by row in
[SEMANTICS.md](../SEMANTICS.md).
