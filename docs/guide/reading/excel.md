# Excel

Spreadsheets are where a lot of science lives. dpyr reads and writes
`.xlsx` workbooks, one sheet at a time.

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

The second argument to `read()` is the **sheet name**. Without it, you
get the first sheet:

```python
from dpyr import read, col

read("field_data.xlsx")                # first sheet
read("field_data.xlsx", "2024 plots")  # the sheet named "2024 plots"
```

Sheet names must match exactly as they appear on the tab in Excel,
including spaces and capitalization.

## Writing

The second argument names the worksheet; without it you get `Sheet1`:

```python
summary.write("results.xlsx")             # one sheet named "Sheet1"
summary.write("results.xlsx", "by_site")  # one sheet named "by_site"
```

Writing replaces the whole file — you can't append a second sheet to an
existing workbook. To ship several tables, write several files, or use
a [duckdb database file](databases.md), which is built for exactly that.

## When things go wrong

- **Wrong sheet name** — the error lists what the workbook actually
  contains, so the fix is right there:

    ```text
    DpyrError: no sheet named 'plot' in 'field_data.xlsx'; sheets in
    this workbook: ['2024 plots', 'notes']
    ```
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
  [parquet](parquet.md) — subsequent reads are faster and typed.
