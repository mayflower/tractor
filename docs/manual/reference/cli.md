# CLI and runtime reference

Run `uv run --no-sync tractor --help` from the configured checkout. Each subcommand
has its own `--help`. The CLI downloads models only when explicitly preparing them;
serving loads a local directory.

## Serving

```sh
uv run --no-sync tractor serve --model models/tractor --device cpu
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | required | Complete local bundle directory. |
| `--device` | `cpu` | `cpu` or `cuda`. |
| `--precision` | `fp32` | `fp32`, `fp16`, or `bf16`; reduced precision requires CUDA. |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8000` | HTTP port. |
| `--max-pending` | `1` | Nonnegative number of waiting requests beyond one active job. |
| `--wait-timeout` | `1.0` | Positive seconds to acquire the inference lock. |

The server runs one worker. The 128 KiB body limit is configurable through Python's
`tractor.server.create_app(body_limit=...)`, not a serving CLI flag. The token limit
comes from the model/encoder configuration; there is no truncate or max-tokens
request flag. The current released bundle has a 512-token joint budget.

## Other workflows

| Command | Purpose and key arguments |
|---|---|
| `prepare-model --output DIR` | Download/prepare upstream base weights for development; default `models/base`. |
| `data generate --output DIR --seed N --sizes TRAIN DEV TEST` | Generate bootstrap synthetic data. |
| `data reliability --output DIR --seed N --sizes TRAIN DEV TEST` | Generate broader synthetic cases; default directory `data/synthetic`. |
| `data audit --input DIR [--model DIR]` | Audit a generated corpus; optional native target checks. |
| `data validate --input DIR --model DIR --device cpu` | Validate data with the model. |
| `data import --input FILE --output FILE` | Validate/import rows; refuses an existing destination. |
| `train --model DIR --data DIR --config FILE [--device cuda]` | Explicit training workflow. |
| `evaluate --model DIR --output DIR [--data DIR] [--split dev]` | Evaluate `dev`, `test`, or `authored`; defaults to CUDA. |
| `export --run DIR --selection FILE [--output DIR]` | Export a selected trained bundle; default output `models/tractor`. |
| `bench --url URL --output DIR` | Measure a running HTTP service; profile `short` or `matrix`. |

`evaluate` also accepts `--device`, `--precision`, `--calibrate`, and `--selection`.
Use only development data to select checkpoints or thresholds; never calibrate on
a final test set. `bench` defaults to 1,000 requests and 100 warmups; `--count` and
`--warmup` override those values. The separate provider comparison runner is
`python -m tractor.compare`; see [its instructions](../../../benchmarks/README.md).

These development commands do not run automatically during serving. Do not overwrite
the released bundle while experimenting. For model training details and compatibility
constraints, see [the architecture](../../architecture.md) and
[upstream compatibility](../../upstream-compatibility.md).

## Environment

`OMP_NUM_THREADS` and `MKL_NUM_THREADS` control CPU library threading. The published
CPU comparison used two threads. The Nix shell configures native library paths,
Python 3.12, and project-local uv/Hugging Face caches. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` prevent Hub loading during inference; the model loader and
container set these for serving. CLI options, not a `TRACTOR_URL` variable, configure
the server. `TRACTOR_URL` is a convenience used by the manual's client examples.

[Manual home](../index.md)
