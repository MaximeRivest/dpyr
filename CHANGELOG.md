# Changelog

Every release of dpyr, newest first. Versions follow
[semantic versioning](https://semver.org). Numbers like S36 point to the
decision table in [docs/SEMANTICS.md](docs/SEMANTICS.md).

## 1.12.0 — 2026-10-03

A list of records becomes rows and columns.

### Added
- `df.unnest_longer(col.x)`: one row per element of a list column, the
  other columns repeated, as tidyr's `unnest_longer()`. A row whose list
  is empty or missing is dropped; `keep_empty=True` keeps it with a null
  element (S39).
- `df.unnest_wider(col.x)`: one column per field of a struct column, in
  its place, as tidyr's `unnest_wider()`. A missing record gives nulls. A
  field named like an existing column is an error, which
  `names_sep="_"` solves by naming the new columns `x_field`.
- `df.unnest(col.x)`: both at once for a list of records, as tidyr's
  `unnest()` of a list of data frames. This is how a row function that
  returns `list[SomeDataclass]` becomes one row per record:
  `papers.mutate(items=extract(col.text)).unnest(col.items)`.
- Both engines, with row order kept: each input row in order, then each
  list in order.

## 1.11.0 — 2026-09-30

`read()` is the one "make this a table" call.

### Added
- `read()` takes back everything it returns. A dpyr dataframe (grouped
  or not) comes back unchanged, like `pd.DataFrame(df)`, so a function
  that accepts "a table or anything like one" can start with
  `papers = read(papers)`. A `Database` or `Workbook` also comes back
  unchanged, and a second argument picks one table or sheet from it:
  `read(db, "orders")`.
- `df.to_dicts()` returns plain Python rows, one dict per row: the
  reverse of `read(list of records)`.

### Changed
- `read([])` returns an empty table with no rows and no columns, as
  `bind_rows(list())` does in R (S36). It used to raise an error. To give
  an empty table column names, pass empty columns:
  `read({"title": [], "score": []})`.
- Every function that takes a file path is typed as accepting
  `pathlib.Path` as well as `str`: `read_parquet`, `read_csv`,
  `read_ipc`, `read_duckdb`, `write`, `write_parquet`, `write_csv`,
  `write_ipc`, `write_duckdb`.

### Fixed
- `write_parquet()` and `write_csv()` crashed when given a
  `pathlib.Path` for a dataframe that runs in duckdb. 1.6.0 promised
  `Path` support everywhere; these two paths had been missed.

## 1.10.1 — 2026-09-26

### Fixed
- A displayed dataframe shows `dpyr.options.preview_rows` rows. Before,
  polars cut every printout to its own 10-row default, so setting
  `preview_rows = 50` still showed 10.

## 1.10.0 — 2026-09-26

Your own functions, row by row.

### Added
- `vectorize(fn)` makes any Python function usable in `mutate()` and
  `filter()`, with columns and constants as arguments (S38). The column
  type comes from the return annotation and is checked when the chain is
  written. One call per distinct input, results remembered for the
  session; a displayed dataframe runs only the rows it shows;
  `threads=` for functions that wait on the network; failures raise
  after every row ran, and `errors="null"` keeps going.
- Objects can decide how they are vectorized (`__dpyr_vectorize__`).
- New guide: *Your own functions, row by row*.

## 1.9.0 — 2026-09-26

Row-shaped data and nested columns. Tagged on GitHub but never reached
PyPI (the publish step failed); everything below first shipped to PyPI
in 1.10.0.

### Added
- `read([{...}, {...}])`: a list of records (dicts, dataclasses,
  namedtuples or pydantic models), one per row, with `bind_rows()`
  semantics. Every row is scanned, so a key or type that first appears
  late is kept (S36).
- List, array and struct columns are carried through verbs on both
  engines, and refused wherever they would be compared (S35).
  `where(is_nested)` selects them.
- `.mean()`, `.std()`, `.var()` on booleans count TRUE as 1, as in R
  (S37).

### Fixed
- In-engine duckdb writes, `to_table()` and `persist()` no longer leak
  internal `__rn` helper columns after `arrange()`.

## 1.8.1 — 2026-06-10

### Changed
- The printout and messages say "dataframe", matching the docs.

## 1.8.0 — 2026-06-10

### Added
- Spreadsheets are catalogs: a multi-sheet `.xlsx` opens as a
  `Workbook`, writes keep the other sheets, and Google Sheets URLs read
  directly.

## 1.7.1 — 2026-06-10

### Added
- A reading guide per format.

### Changed
- A bad sheet name lists the workbook's sheets.

## 1.7.0 — 2026-06-10

### Added
- Joins work across database connections (a second duckdb file, a
  sqlite file, separate in-memory databases): the foreign side streams
  through Arrow, with a warning naming the table so large copies stay
  visible.

## 1.6.0 — 2026-06-10

### Added
- A format registry: each format is a self-contained module, and
  `read()`/`write()` dispatch over it.
- New formats: `.json`, `.jsonl`/`.ndjson`, `.tsv`, `.csv.gz`, `.xlsx`
  (`pip install 'dpyr[excel]'`), and `.sqlite` read through duckdb.
- `read()`/`write()` accept `pathlib.Path`.
- New *Reading & writing* guide.

## 1.5.0 — 2026-06-10

### Added
- `read()` takes Hugging Face datasets (`read(dataset_dict, "train")`
  picks a split), `hf://` paths, numpy arrays, and torch/jax tensors.
- `to_numpy()`, `to_torch()`, `to_jax()`.

## 1.4.0 — 2026-06-10

### Added
- `read()` takes dicts, polars/pandas dataframes, arrow tables and live
  duckdb connections as well as paths.

### Changed
- The docs teach two IO words, `read()` and `df.write()`; the
  format-specific functions stay as escape hatches.

## 1.3.0 — 2026-06-10

### Added
- `read(path)` and `df.write(path)` dispatch on the file extension;
  duckdb files open as a catalog.
- `write_csv()`.

## 1.2.0 — 2026-06-10

### Added
- In-memory dataframes join duckdb tables directly (the plan runs in
  duckdb, which reads the Arrow data in place); `collect(engine=)`.
- `to_table()`, `to_view()`, `write_duckdb()`, `write_parquet()`,
  `write_ipc()`, `show_query()`, `read_duckdb()`, `read_ipc()`,
  `glimpse()`.

### Changed
- `persist()` on duckdb runs fully in-engine.
- `slice_sample()` picks the same rows for the same seed on both
  engines (S33).

## 1.1.0 — 2026-06-10

### Added
- Window functions: `lag`/`lead`, `row_number`, `min_rank`,
  `dense_rank`, `percent_rank`, `cum_sum`/`cum_min`/`cum_max`.
- `slice_min`/`slice_max`, `separate`/`unite`/`relocate`,
  `coalesce`/`replace_na`.

### Fixed
- A rare nondeterministic slice result on duckdb.

## 1.0.0 — 2026-06-10

First stable release: dplyr's verbs as Python method chains on polars
and duckdb, checked against real dplyr with golden tests, with typed
autocompletion.
