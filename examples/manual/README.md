# Runnable manual examples

Start a ready tractor server first; examples default to `http://127.0.0.1:8000`.
Set `TRACTOR_URL` to use another address. Run Python commands inside `nix develop`
from the repository root, with `just setup` already completed.

| File | Use |
|---|---|
| `scalar.json` | curl request: German width converted to mm. |
| `records.json` | curl/fetch request: two products with different source units. |
| `python_client.py` | SDK records, Decimal, missing values, and Unicode evidence. |
| `async_http.py` | Reusable async HTTP client with bounded concurrency. |
| `node.mjs` | Node fetch integration that preserves raw JSON decimals. |
| `batch.py` | JSONL import with per-item errors and review routing. |
| `requests.jsonl` | Sample batch source. |
| `sqlite_import.py` | Idempotently store accepted batch values as decimal text. |
| `cases.jsonl` | Expected results for 15 published model integration examples. |
| `check_examples.py` | Verify those examples against the running release API. |
| `compose.yaml` | Local CPU deployment with a read-only model mount. |

```sh
uv run --no-sync python examples/manual/python_client.py
uv run --no-sync python examples/manual/async_http.py
node examples/manual/node.mjs
uv run --no-sync python examples/manual/batch.py examples/manual/requests.jsonl /tmp/tractor-results.jsonl
uv run --no-sync python examples/manual/sqlite_import.py /tmp/tractor-results.jsonl /tmp/tractor-products.sqlite
uv run --no-sync python examples/manual/check_examples.py
```

Choose a new JSONL output path for another batch run. Read the
[user manual](../../docs/manual/index.md) for expected outputs and integration limits.
