# Excel & Google Sheets

Spreadsheets are where a lot of science lives. dpyr treats an `.xlsx`
workbook like a small database: one sheet reads as a frame, several
sheets open as a catalog, and writing adds or replaces one sheet at a
time without destroying the rest. Google Sheets work the same way —
paste the browser URL into `read()`.

## Setup (one time)

Excel support is an optional extra — install it once:

```bash
pip install 'dpyr[excel]'        # or: uv pip install 'dpyr[excel]'
```

If you forget, dpyr tells you exactly that:

```text
DpyrError: reading .xlsx needs the excel extra: pip install 'dpyr[excel]'
```

## Reading

A workbook with **one** sheet behaves like a CSV — you get the frame:

```python
from dpyr import read, col

field = read("field_data.xlsx")          # the single sheet, as a frame
```

A workbook with **several** sheets opens as a `Workbook` catalog, the
same shape as a [database](databases.md): sheets are attributes, with
tab completion and did-you-mean on typos.

```python
wb = read("report.xlsx")
print(wb)
```

```text
# excel workbook: report.xlsx
#   2024 plots
#   notes
# read a sheet: wb.sheet(name), wb[name], or wb.<name>
```

```python
wb.sheets                  # ['2024 plots', 'notes']
wb.notes                   # plain names work as attributes
wb["2024 plots"]           # names with spaces use [...] or .sheet(...)
```

To skip the catalog, name the sheet directly as `read()`'s second
argument — and a wrong name tells you what the workbook holds:

```python
read("report.xlsx", "2024 plots")
read("report.xlsx", "2024 plot")
# DpyrError: no sheet named '2024 plot' in 'report.xlsx'.
# Did you mean '2024 plots'? Sheets: 2024 plots, notes
```

Sheet names must match exactly as they appear on the tab in Excel,
including spaces and capitalization.

## Writing

The second argument names the worksheet; without it you get `Sheet1`:

```python
summary.write("results.xlsx")             # one sheet named "Sheet1"
summary.write("results.xlsx", "by_site")  # one sheet named "by_site"
```

Writing into an **existing** workbook replaces only the named sheet and
keeps the others, so building a multi-sheet report is just repeated
writes:

```python
plots.write("report.xlsx", "plots")
notes.write("report.xlsx", "notes")       # plots sheet survives
plots2.write("report.xlsx", "plots")      # replaced in place, notes survives
```

One honest caveat, and dpyr warns about it: the carried-over sheets are
rewritten from their *values*, so any hand-applied cell formatting
(colors, column widths, formulas) in them is lost. For
formatting-heavy workbooks, write your data to its own file and link it
from the formatted one.

## Google Sheets

Copy the sheet's URL straight from the browser's address bar — any
`docs.google.com/spreadsheets/...` link works, no API keys and no
client libraries:

```python
wb = read("https://docs.google.com/spreadsheets/d/1AbC.../edit?gid=0")
wb.sheets                          # same Workbook catalog as a local file
read("https://docs.google.com/spreadsheets/d/1AbC.../edit", "plots")
```

The whole workbook is fetched through Google's export endpoint, so
sheet selection, the catalog, and the error messages are identical to
local Excel. Two things to know:

- **The sheet must be link-readable.** In Google Sheets: Share →
  "Anyone with the link" (Viewer). A private sheet fails with exactly
  that instruction — dpyr can't log into your Google account:

    ```text
    DpyrError: this Google Sheet is not link-readable: '...'. In Google
    Sheets use Share -> 'Anyone with the link' (Viewer), or
    File -> Download -> .xlsx and read the file
    ```

- **Reads are a snapshot.** Each `read()` downloads the current state
  of the sheet; collaborators' later edits appear on the next `read()`.

There is no writing back to Google Sheets — write an `.xlsx` and
import it, or keep results in [parquet](parquet.md)/[duckdb](databases.md).

## When things go wrong

- **Missing file** — `DpyrError: read('field_data.xlsx'): no such file`.
  Check the spelling and your working directory.
- **Wrong sheet name** — the error names the available sheets and
  suggests the closest match (see above).
- **Numbers come in as text** (or dates as numbers) — usually the
  spreadsheet itself has mixed content in the column, often a stray
  note typed into a cell. Clean the column in Excel, or cast after
  reading: `.mutate(x=col.x.cast(float))`.
- **Headers aren't on row 1** — dpyr expects the column names in the
  first row. Title rows and merged cells above the data confuse the
  reader; delete them in Excel first.

## Good to know

- Excel files are read eagerly (the whole sheet is parsed up front),
  unlike CSV and parquet which are lazy scans. Fine for the sizes Excel
  handles anyway.
- If a workbook is a one-time import, read it once and `.write()` to
  [parquet](parquet.md) — subsequent reads are faster and typed. For a
  many-table result that stays in the data world, a
  [duckdb file](databases.md) is the better container; reach for
  multi-sheet Excel when the audience is a human with a spreadsheet
  program.
