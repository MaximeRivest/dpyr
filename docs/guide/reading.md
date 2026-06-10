# Reading & writing

Two words cover every way in and out of dpyr: `read()` and `.write()`.
`read()` takes a path, a URL, or any tabular object in memory and gives
you a frame (or a database catalog); `.write()` sends a result to any of
the same places. Whatever the source, the data ends up as Arrow in RAM,
and your verbs run on polars or duckdb — you never pick a parser.

## One table of everything

| Source / destination | `read()` | `.write()` | Notes |
|---|---|---|---|
| `.parquet` / `.pq` | ✓ | ✓ | lazy scan; globs work; duckdb plans COPY in-engine |
| `.csv` | ✓ | ✓ | lazy scan; in-engine COPY on duckdb |
| `.csv.gz` | ✓ | — | decompressed eagerly |
| `.tsv` / `.tsv.gz` | ✓ | ✓ (`.tsv`) | tab-separated |
| `.json` | ✓ | ✓ | a single JSON document |
| `.jsonl` / `.ndjson` | ✓ | ✓ | newline-delimited; lazy scan; in-engine COPY |
| `.arrow` / `.feather` / `.ipc` | ✓ | ✓ | memory-mapped on read |
| `.xlsx` | ✓ | ✓ | needs `pip install 'dpyr[excel]'`; sheet via second argument |
| `.db` / `.duckdb` / `.ddb` | ✓ | ✓ | a database: catalog on read, named table on write |
| `.sqlite` / `.sqlite3` | ✓ | — | read through duckdb's sqlite scanner |
| `hf://`, `s3://`, `https://` URLs | ✓ | — | suffix decides the format, with pushdown |
| dict, polars, pandas, arrow | ✓ | n/a | in-memory objects, zero/near-zero copy |
| Hugging Face datasets | ✓ | n/a | arrow-backed; splits via `read(dd, "train")` |
| numpy / torch / jax | ✓ | n/a | and back out: `to_numpy()` / `to_torch()` / `to_jax()` |
| live duckdb connection | ✓ | ✓ (`to_table`) | `read(con)` → catalog |

An unknown extension fails with the list of what's supported, so the
error message is also the documentation.

## Files round-trip

```python
import tempfile, pathlib
from dpyr import read, col, n

tmp = pathlib.Path(tempfile.mkdtemp())

trees = read({
    "species": ["sugar maple", "red oak", "white pine", "sugar maple"],
    "height_m": [24.0, 19.5, 31.0, 12.5],
    "tapped":  [True, False, False, True],
})

trees.write(str(tmp / "trees.parquet"))
trees.write(str(tmp / "trees.jsonl"))
trees.write(str(tmp / "trees.tsv"))

tall = read(str(tmp / "trees.parquet")).filter(col.height_m > 15)
print(tall.collect()["species"].to_list())
print(read(str(tmp / "trees.jsonl")).collect().height)
```

```text
['sugar maple', 'red oak', 'white pine']
4
```

File reads are *scans*: nothing is parsed until something materializes,
and only the columns and rows your chain needs are touched. Arrow IPC
goes further — `read("x.arrow")` memory-maps, so opening a 10 GB file is
instant.

## Databases are files too

A duckdb file is the modern "folder of CSVs": one file, many typed
tables, instant open. Reading one gives a catalog whose attributes are
frames; writing needs a table name:

```python
summary = trees.group_by(col.tapped).summarize(n=n(), tallest=col.height_m.max())
summary.write(str(tmp / "forest.db"), "tap_summary")

db = read(str(tmp / "forest.db"))
print(db.tables)
print(db.tap_summary.filter(col.tapped).collect().to_dicts())
```

```text
['tap_summary']
[{'tapped': True, 'n': 2, 'tallest': 24.0}]
```

Legacy SQLite files open the same way (read-only, through duckdb's
sqlite scanner — the first use downloads the extension):

```python
import sqlite3
sq = sqlite3.connect(str(tmp / "legacy.sqlite"))
sq.execute("CREATE TABLE plots (plot TEXT, acres REAL)")
sq.execute("INSERT INTO plots VALUES ('north', 3.2), ('south', 1.8)")
sq.commit(); sq.close()

plots = read(str(tmp / "legacy.sqlite"), "plots")
print(plots.filter(col.acres > 2).collect().to_dicts())
```

```text
[{'plot': 'north', 'acres': 3.2}]
```

A live `duckdb.connect()` connection works like a database file:
`read(con)` is the catalog, `read(con, "orders")` one table, and
`read(con).sql("SELECT ...")` runs arbitrary SQL.

## Remote data

The suffix decides the format for URLs exactly as for local paths, and
both engines push column selections and filters into the request:

```text
read("https://example.org/events.parquet")     # any HTTP(S) host
read("s3://bucket/logs/*.parquet")             # object stores
read("hf://datasets/user/dataset/data.parquet") # the Hugging Face Hub
```

(Cloud credentials come from the usual environment variables —
`AWS_*`, `HF_TOKEN`.)

## In-memory objects

Everything tabular that's already in your process goes through the same
door — dicts, polars and pandas frames, Arrow tables, Hugging Face
datasets, numpy arrays, and CPU torch/jax tensors. The
[backends guide](backends.md#ml-data-hugging-face-datasets-numpy-tensors)
shows the ML side in detail, including the `to_numpy()` / `to_torch()` /
`to_jax()` exits.

## Where write() runs

On duckdb-backed chains, `write()` to parquet, csv, or jsonl compiles to
an in-engine `COPY (<query>) TO ...` — the rows go straight from the
database to the file without entering Python. To land results *inside*
an engine instead of a file, see `to_table()` and `to_view()` in the
[backends guide](backends.md#landing-results-in-the-engine).
