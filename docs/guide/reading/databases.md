# Databases

A duckdb file is the modern "folder of CSVs": one file, many typed
tables, instant open, no server and no login. If your project has
outgrown a pile of spreadsheets, this is the next step — and dpyr
treats the file like any other path.

## duckdb files (`.db`, `.duckdb`, `.ddb`)

A database holds many tables, so this is where `read()`'s second
argument earns its keep — it names the table:

```python
from dpyr import read, col, n

db = read("forest.db")            # the whole catalog
plots = read("forest.db", "plots")  # one table, as a frame
```

Reading the catalog gives a `Database` object: `db.tables` lists the
table names, and each table is an attribute —

```python
print(db.tables)                  # ['plots', 'tap_summary']
db.plots.filter(col.acres > 2).collect()
```

Printing `db` shows every table with its columns and types, so when you
don't know what's inside a file, `print(read("mystery.db"))` is the
answer.

### Writing

Writing into a database **requires** a table name, and dpyr reminds you
if you forget:

```python
summary.write("forest.db", "tap_summary")   # creates/replaces the table
summary.write("forest.db")
# DpyrError: write('forest.db') needs a table name: write('forest.db', 'orders')
```

The file is created if it doesn't exist, so
`dataframe.write("new.db", "results")` is all it takes to start a database.

## SQLite files (`.sqlite`, `.sqlite3`)

Lots of instruments and older tools hand you SQLite. dpyr opens these
read-only through duckdb's sqlite scanner — same shape as duckdb files:

```python
db = read("legacy.sqlite")               # catalog
users = read("legacy.sqlite", "users")   # one table
```

The first use downloads duckdb's sqlite extension, so it needs network
access once. If the file isn't actually SQLite (or the extension can't
load), the error says so:

```text
DpyrError: could not open 'legacy.sqlite' as sqlite via duckdb's sqlite extension: ...
```

There's no SQLite writer — write to a duckdb file instead.

## Live connections

If you already hold a `duckdb.connect()` connection — in-memory work, a
database with attached remotes, custom settings — `read()` takes it
directly:

```python
import duckdb
con = duckdb.connect("forest.db")

db = read(con)                    # catalog, like read("forest.db")
orders = read(con, "orders")      # one table
db.sql("SELECT * FROM plots WHERE acres > 2")   # arbitrary SQL -> frame
```

Connections to *server* databases (Postgres, MySQL) aren't directly
supported — but duckdb can `ATTACH` them, and `read(con)` then sees
their tables. The [backends guide](../backends.md) covers landing
results inside an engine with `to_table()` and `to_view()`.

## When things go wrong

- **`read_duckdb: no such file 'forest.db'`** — reading never creates a
  database (writing does). Check the path.
- **Unknown table name** — `print(read("file.db"))` lists what's
  actually in there; names are case-sensitive.
- **Joining tables from two different database files** — dataframes from
  different connections can't meet in one query; read both tables from
  the same file/connection, or see [Joins](../joins.md) for how dpyr
  bridges across.
