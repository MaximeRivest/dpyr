# JSON

Two different things share the name, and the extension tells them apart:

- **`.json`** — one JSON document, typically a list of records:
  `[{"plot": "north", "acres": 3.2}, ...]`
- **`.jsonl`** / **`.ndjson`** — one JSON object *per line*. This is
  what logging systems and APIs usually emit, and it's the better
  format for data: it can be scanned lazily, line by line.

## Reading

```python
from dpyr import read, col

read("plots.json")        # a single document, parsed up front
read("events.jsonl")      # newline-delimited, lazy scan
read("events.ndjson")     # same thing, other common extension
```

No second argument for either — each file is one table. Nested fields
arrive as struct/list columns, which work but are nicer to flatten
early in the chain.

## Writing

```python
result.write("plots.json")     # one document: a JSON array of objects
result.write("events.jsonl")   # one object per line
```

## When things go wrong

- **`read() can't infer a format from 'data.json5'`** — only `.json`,
  `.jsonl`, and `.ndjson` are recognized; the error lists every
  readable extension.
- **A `.json` file that's actually line-delimited** (or vice versa) —
  the parse error will look cryptic. Peek at the file: starts with `[`
  → it's `.json`; one `{...}` per line → rename to `.jsonl`.
- **Inconsistent records** (a field present in some objects, missing in
  others) — missing fields become nulls, which is usually what you
  want. Records whose *types* disagree (`"x": 1` then `"x": "high"`)
  fail the parse; fix the producer or pre-clean the file.

## Good to know

- `.jsonl` scans lazily and, on duckdb, writes in-engine via `COPY` —
  it behaves like CSV with better types. `.json` documents are read
  eagerly, all at once.
- For data you'll re-read, [parquet](parquet.md) remains the better
  resting place.
