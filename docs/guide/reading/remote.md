# Remote data

URLs work exactly like local paths: the suffix decides the format, and
the same `read()` does the work.

```python
from dpyr import read, col

read("https://example.org/events.parquet")        # any HTTP(S) host
read("s3://bucket/logs/*.parquet")                # object stores
read("hf://datasets/user/dataset/data.parquet")   # the Hugging Face Hub
read("https://docs.google.com/spreadsheets/d/...")  # a Google Sheet
```

(Google Sheets URLs have their own page — see
[Excel & Google Sheets](excel.md#google-sheets).)

For parquet, both engines push your column selections and filters into
the request itself — filtering a huge remote file downloads only the
byte ranges that survive the filter, not the whole file.

## Logging in (credentials)

There are no login arguments in dpyr. Credentials come from the
standard environment variables of each service, which means they work
the same here as in every other tool:

- **Public HTTP(S)** — nothing to do.
- **S3 and S3-compatible stores** — the usual `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (an `aws configure`
  setup is picked up too).
- **Hugging Face** — public datasets just work; gated/private ones need
  `HF_TOKEN` set, or a one-time `huggingface-cli login`.

Set them in your shell before starting Python:

```bash
export HF_TOKEN=hf_...
python analysis.py
```

(or put them in a `.env` your environment loads — never paste secrets
into the script itself.)

## Hugging Face datasets

Besides `hf://` parquet URLs, an already-loaded `datasets` object goes
straight in, with the split as the second argument:

```python
from datasets import load_dataset
dd = load_dataset("user/dataset")
train = read(dd, "train")
```

See [In-memory objects](in-memory.md) for the rest of the ML side.

## When things go wrong

- **403 / 401 errors** — the credential variables aren't set in *this*
  process. Check with `import os; os.environ.get("HF_TOKEN")`.
- **Slow first touch** — remote reads are lazy like local ones; the
  download happens at `collect()`, not at `read()`. Select the columns
  you need early to shrink it.
- **A URL without a recognizable suffix** — dpyr can't guess the
  format; the error lists the readable extensions. If the URL hides the
  format behind a query string, download it to a named file first.
