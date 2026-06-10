# Reading & writing

Two words cover every way in and out of dpyr: `read()` and `.write()`.
`read()` takes a path, a URL, or any tabular object in memory and gives
you a frame (or a database catalog); `.write()` sends a result to any of
the same places. Whatever the source, the data ends up as Arrow in RAM,
and your verbs run on polars or duckdb — you never pick a parser.

There are deliberately almost no options to learn. `read()` takes the
*thing* (a path, a URL, an object) and, when the source contains more
than one table, a *name* — the sheet of a spreadsheet, the table of a
database, the split of a dataset:

```python
read("trees.csv")             # one-table sources: just the path
read("survey.xlsx", "2024")   # the sheet called "2024"
read("forest.db", "plots")    # the table called "plots"
```

Multi-table sources opened *without* a name give you a catalog you can
explore — a duckdb file lists its tables, a multi-sheet workbook its
sheets — so `print(read("mystery.xlsx"))` is always a safe first move.

That's the whole API. Everything format-specific — what the second
argument means, what can go wrong, how to fix it — lives on one page per
format below.

## Every source and destination

| Source / destination | `read()` | `.write()` | Details |
|---|---|---|---|
| `.csv`, `.tsv` (and `.gz`) | ✓ | ✓ | [CSV & TSV](csv.md) |
| `.parquet` / `.pq` | ✓ | ✓ | [Parquet](parquet.md) |
| `.xlsx`, Google Sheets URLs | ✓ | ✓ (`.xlsx`) | [Excel & Google Sheets](excel.md) |
| `.json`, `.jsonl` / `.ndjson` | ✓ | ✓ | [JSON](json.md) |
| `.arrow` / `.feather` / `.ipc` | ✓ | ✓ | [Arrow IPC](arrow.md) |
| `.db` / `.duckdb` / `.ddb`, `.sqlite` / `.sqlite3`, live connections | ✓ | ✓ | [Databases](databases.md) |
| `https://`, `s3://`, `hf://` URLs | ✓ | — | [Remote data](remote.md) |
| dict, polars, pandas, arrow, numpy, torch/jax, 🤗 datasets | ✓ | n/a | [In-memory objects](in-memory.md) |

An unknown extension fails with the list of what's supported, so the
error message is also the documentation. And every source joins every
other source — see [Joins](../joins.md).

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
and only the columns and rows your chain needs are touched.

## Where write() runs

On duckdb-backed chains, `write()` to parquet, csv, or jsonl compiles to
an in-engine `COPY (<query>) TO ...` — the rows go straight from the
database to the file without entering Python. To land results *inside*
an engine instead of a file, see `to_table()` and `to_view()` in the
[backends guide](../backends.md#landing-results-in-the-engine).
