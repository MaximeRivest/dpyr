# CSV & TSV

The format your collaborators email you. dpyr reads and writes both
comma-separated (`.csv`) and tab-separated (`.tsv`) files, plus their
gzip-compressed variants.

## Reading

```python
from dpyr import read, col

field = read("field_measurements.csv")
field.filter(col.height_m > 15).collect()
```

No options needed: the delimiter comes from the extension (`.csv` →
comma, `.tsv` → tab), the header row is detected, and column types are
inferred from the data. The second argument to `read()` doesn't apply
here — a CSV is always exactly one table.

Reading is a *scan*: the file isn't parsed until you `collect()`, and
only the columns your chain actually uses are read. A 2 GB CSV filtered
down to one species costs far less than 2 GB of work.

### Compressed files

`.csv.gz` and `.tsv.gz` open exactly the same way:

```python
read("survey_2019.csv.gz")
```

One difference: compressed files are decompressed up front rather than
scanned lazily, so very large `.gz` files use memory proportional to
their full size. If a file is big enough for that to hurt, convert it
once to [parquet](parquet.md) and work from that.

## Writing

```python
result.write("summary.csv")
result.write("summary.tsv")
```

Writing always includes a header row. There is no `.csv.gz` writer —
write a plain `.csv`, or better, a [parquet file](parquet.md), which is
smaller than gzipped CSV *and* keeps the column types.

## When things go wrong

- **File doesn't exist** — the error names the missing path. Check the
  spelling and your working directory (`import os; os.getcwd()`).
- **Columns read as the wrong type** (e.g. an ID column of `00123`
  codes becomes an integer) — type inference looks at the data, and
  ambiguous columns can guess wrong. Fix the type in the chain:
  `read("f.csv").mutate(id=col.id.cast(str))`.
- **A non-standard delimiter** (semicolons, pipes) — dpyr trusts the
  extension. For exotic files, parse with polars directly and hand the
  result to dpyr: `read(pl.read_csv("odd.txt", separator=";"))`.

## Good to know

- CSV stores no types — every read re-infers them, and dates often
  arrive as text. If you read the same file more than twice, write it
  to [parquet](parquet.md) once and read that instead: faster, smaller,
  and the types stick.
- On the duckdb backend, `.write("out.csv")` runs inside the database
  engine (`COPY ... TO`), so big results never pass through Python.
