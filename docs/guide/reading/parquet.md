# Parquet

The format to standardize on. Parquet files are compressed, store the
column types (dates stay dates), and can be read partially — dpyr only
touches the columns and rows your chain needs. If your project reads
the same data many times, convert it to parquet once:

```python
read("survey.csv").write("survey.parquet")
```

## Reading

```python
from dpyr import read, col

trees = read("trees.parquet")        # .pq works too
trees.filter(col.species == "sugar maple").collect()
```

No options, no second argument — a parquet file is one table, complete
with its schema. Reads are lazy scans: opening the file is instant, and
work happens at `collect()`.

### Many files at once

Globs work, which is how partitioned datasets are usually stored:

```python
read("plots/*.parquet")          # every file in the folder, one frame
read("logs/2026-*/**.parquet")   # nested folders too
```

All files must share the same columns; they're read as one big table.

## Writing

```python
result.write("summary.parquet")
```

On the duckdb backend this compiles to an in-engine
`COPY (<query>) TO ... (FORMAT PARQUET)` — the rows go straight from
the database to the file. On polars it streams, so results larger than
RAM can still be written.

## When things go wrong

- **File doesn't exist** — the error names the path; check spelling and
  working directory.
- **`read(table=...) does not apply to parquet files`** — you passed a
  second argument. Parquet has no sheets or tables; just the path.
- **Glob matches nothing** — you'll get an empty-source error from the
  engine. Test the pattern with `import glob; glob.glob("plots/*.parquet")`.

## Good to know

- Parquet + dpyr is the lazy combination: `read()` costs nothing, and
  filters and column selections are pushed into the file read itself.
- This is also the best format for [remote data](remote.md) — over
  HTTP or S3, only the needed byte ranges are downloaded.
