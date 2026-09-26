# Your own functions, row by row

dpyr's expressions compile to polars or duckdb, which is why they are fast.
Some work has no expression, though: a regular Python function you already
have, a parser, an API call, a language model. `vectorize()` turns any such
function into something you can use inside `mutate()` and `filter()`:

```python
from dataclasses import dataclass
from dpyr import read, col, vectorize

reviews = read([
    {"id": 1, "product": "kettle", "text": "Boils fast, loud click."},
    {"id": 2, "product": "kettle", "text": "Broke after a week!"},
    {"id": 3, "product": "toaster", "text": "Even toast, nice design."},
    {"id": 4, "product": "toaster", "text": "Broke after a week!"},
])

@vectorize
def word_count(text: str) -> int:
    return len(text.split())

reviews.mutate(words=word_count(col.text))
```

The name comes from `numpy.vectorize`, and the warning is the same: this is
a convenience, not a speed-up. The function runs in Python, once per row.
Use a dpyr expression whenever one exists (`col.text.str_len()` is far
faster than a Python `len`).

## Several inputs, and constants

Pass columns and plain values, by position or by name. A column gives each
row its own value; anything else is the same on every row:

```python
@vectorize
def tag(product: str, text: str, marker: str = "!") -> str:
    return f"{product}: {'complaint' if marker in text else 'ok'}"

reviews.mutate(
    label=tag(col.product, col.text),
    strict=tag(col.product, text=col.text, marker="Broke"),
)
```

Arguments can be any expression, and the result can feed other
expressions, in the same `mutate()`:

```python
reviews.mutate(long=word_count(col.text) > 4)
reviews.filter(word_count(col.text) > 4)
```

## The column's type

The column type comes from the function's return annotation, so a chain is
still checked on the line you write it: `word_count(col.text) + "x"` fails
immediately, before anything runs. Annotations can be `str`, `int`,
`float`, `bool`, dates, an `Enum` or `Literal` (stored as their values),
`Optional[...]`, `list[...]`, or a dataclass, pydantic model or TypedDict
(a struct column). A function without an annotation needs `dtype=`:

```python
@dataclass
class Sentiment:
    label: str
    score: float

@vectorize
def sentiment(text: str) -> Sentiment:
    bad = "Broke" in text
    return Sentiment("negative" if bad else "positive", 0.1 if bad else 0.9)

first_letter = vectorize(lambda text: text[:1], dtype=str)

reviews.mutate(s=sentiment(col.text), initial=first_letter(col.text))
```

A value that doesn't fit the declared type is an error naming the row.

## What dpyr does for you

- **One call per distinct input.** Rows 2 and 4 above have the same text,
  so `word_count` runs 3 times for 4 rows.
- **Results are remembered** for the session. Displaying a dataframe, then
  collecting it, then building another chain on the same column never calls
  the function twice for the same arguments. (`dpyr.cache_clear()` forgets
  them; `vectorize(fn, version="2")` keeps a changed function's results
  apart.)
- **Displaying runs only the shown rows** when the steps after the function
  keep rows one-for-one (`mutate`, `select`, `rename`). A `filter()` or
  `arrange()` after it needs every row, so display runs them all.
- **`threads=`** runs rows concurrently, for functions that wait on the
  network: `@vectorize(threads=8)`.
- **Failures**: by default every row runs, then an error says how many
  failed and why. The results that worked are kept, so running the same
  line again retries only the failed rows. With `errors="null"` the failed
  rows are left missing, with a warning.

```python
calls = []

@vectorize
def slow_upper(text: str) -> str:
    calls.append(text)
    return text.upper()

loud = reviews.mutate(loud=slow_upper(col.text))
loud.collect()
loud.collect()
assert len(calls) == 3          # 4 rows, 3 distinct texts, collected twice
```

## Where it runs

Row functions go in `mutate()` and `filter()` (grouped or not). To
summarize or sort by their result, compute it in `mutate()` first:

```python
(reviews
    .mutate(words=word_count(col.text))
    .group_by(col.product)
    .summarize(mean_words=col.words.mean()))
```

On duckdb, the rows the function needs are read out of the engine, the
function runs in Python, and the result goes back as an in-memory table,
so the steps after it still run in duckdb. `show_query()` can't show a
chain that runs Python in the middle; `collect()` runs it.
