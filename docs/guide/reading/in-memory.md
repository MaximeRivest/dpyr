# In-memory objects

Everything tabular that's already in your Python session goes through
the same door. `read()` dispatches on the object's type — no flags:

```python
from dpyr import read, col

read({"x": [1, 2], "y": ["a", "b"]})   # plain Python data
read(pandas_dataframe)                  # near-zero copy
read(polars_dataframe)                  # zero copy
read(arrow_table)                       # zero copy
read(numpy_2d_array)                    # columns column_0, column_1, ...
read(torch_or_jax_tensor)               # CPU tensors
read(hf_dataset)                        # Hugging Face, arrow-backed
```

This is the everyday bridge from other libraries: whatever a colleague's
pandas script produces, `read()` it and continue in dpyr verbs.

## Dicts: the quick way to test something

A dict of lists is the fastest table you can type, which makes it ideal
for trying a verb or building a tiny example:

```python
trees = read({
    "species": ["sugar maple", "red oak"],
    "height_m": [24.0, 19.5],
})
```

Keys become column names; lists must all have the same length (the
error tells you which one doesn't match).

## Hugging Face datasets

A `DatasetDict` has named splits, so — as everywhere in dpyr where a
source holds several tables — the second argument picks one:

```python
from datasets import load_dataset
dd = load_dataset("user/dataset")
train = read(dd, "train")
```

A single `Dataset` (already one split) needs no second argument.

## Getting back out

`collect()` gives a polars DataFrame, and a dataframe also exits directly
with `to_pandas()`, `to_numpy()`, `to_torch()`, and `to_jax()`. The
[backends guide](../backends.md#ml-data-hugging-face-datasets-numpy-tensors)
shows the ML round-trips in detail.

## When things go wrong

- **`read() doesn't know what to do with <SomeType>`** — the object
  isn't one of the supported kinds; the error lists them. Convert to a
  dict or pandas/polars dataframe first.
- **`read(table=...) only applies to database sources and Hugging Face
  dataset splits`** — the second argument means nothing for a dict or a
  dataframe; drop it.
- **GPU tensors** — move them to CPU first (`tensor.cpu()`); dpyr reads
  CPU memory.
