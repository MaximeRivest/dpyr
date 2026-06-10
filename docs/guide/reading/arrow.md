# Arrow IPC (Feather)

Arrow IPC — `.arrow`, `.feather`, or `.ipc`, all the same format — is
the fastest way to pass tables between tools that speak Arrow (polars,
pandas, R's `{arrow}`, duckdb...). It's dpyr's in-memory format written
straight to disk, so reading one costs almost nothing.

## Reading

```python
from dpyr import read

read("checkpoint.arrow")
```

No second argument — one file, one table. The file is **memory-mapped**:
opening a 10 GB file is instant, and pages are pulled from disk only as
your chain touches them.

## Writing

```python
result.write("checkpoint.arrow")
```

Streams on polars when the plan allows, so larger-than-RAM results work.

## When to choose it

- **Checkpoints and scratch files** within a project — fastest
  round-trip there is.
- **Handing data to R colleagues** — `arrow::read_feather()` reads it
  natively, types intact.
- For long-term storage or sharing over a network, prefer
  [parquet](parquet.md): much smaller on disk, equally typed.

## When things go wrong

- **File doesn't exist / wrong extension** — same rules as everywhere:
  the extension picks the reader, and unknown extensions fail with the
  list of supported ones.
- **Old Feather v1 files** can fail to open; re-export them as v2
  (every current tool writes v2).
