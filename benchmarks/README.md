# Reproduce the extraction comparison

`acceptance.jsonl` contains 680 frozen synthetic requests and their expected
responses, selected before the comparison from the 2,048-request release acceptance
set. The sample spans 38 published scenario labels, with 341 German and 339 English
requests. Source texts are disjoint from training and dev; template families may
be shared. This is a specialized synthetic workload, not a general model ranking.

Selection uses scenario/language round robin with SHA-256 ID ordering. Public IDs
are assigned after selection; the runner establishes one identical order for all
providers. The sample, model weights, prompt and runner are hash-identified in the
run manifests. Two API connectivity pilot requests per cloud provider are excluded
from the main results. Neither model weights nor prompt were adjusted after pilots.

## Run

Install the environment with `nix develop` and `just setup`. Start the released
model on CPU in another terminal:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 tractor serve --model models/tractor --device cpu --port 8017
```

The comparison requires `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and, if required by
your Anthropic account, `ANTHROPIC_WORKSPACE_ID` in the environment. Never commit
credentials. Every output directory must be new. Cloud commands incur API charges.

```sh
uv run --no-sync python -m tractor.compare --provider tractor \
  --data benchmarks/acceptance.jsonl --limit 680 \
  --model-sha256 209a62ad03f559d7fd21fca9e44c4daad597825b1b6520df72545823bbc9c61b \
  --url http://127.0.0.1:8017 --output reports/comparison-tractor
uv run --no-sync python -m tractor.compare --provider anthropic \
  --data benchmarks/acceptance.jsonl --limit 680 --output reports/comparison-haiku
uv run --no-sync python -m tractor.compare --provider openai \
  --data benchmarks/acceptance.jsonl --limit 680 --output reports/comparison-luna
uv run --no-sync python -m tractor.comparison_report \
  --tractor reports/comparison-tractor --haiku reports/comparison-haiku \
  --luna reports/comparison-luna --output benchmarks/results
```

For a new run, optionally pass `--environment` with a JSON description of your
actual hardware. The published `environment.json` describes the captured run only.

The run uses one outstanding request per provider, no retries, no answer repair,
no explicit warmups, a 120-second timeout and 4,096 maximum output tokens.
Haiku thinking and Luna reasoning are disabled. Both use `prompt.txt`; provider
structured-output modes, tool use and provider-specific prompt tuning are disabled.
These are direct prompt-only extraction baselines, not optimized LLM pipelines.

## Inspect and rescore

`results/` contains sanitized per-request responses, usage and timings, plus each
run manifest and `summary.json`. Provider request IDs, credentials and HTTP headers
are excluded. Gold responses are never sent to providers. Rescore the captured
results without making API calls:

```sh
uv run --no-sync python -m tractor.comparison_report \
  --tractor benchmarks/results/tractor --haiku benchmarks/results/haiku \
  --luna benchmarks/results/luna --environment benchmarks/environment.json \
  --output /tmp/tractor-rescored
```

Complete-response exact match includes record order, all fields, JSON numbers and
issue codes/order. Data-only exact match ignores the issues array. Numeric-tolerant
scores preserve the same structure, strings and record order but allow numeric
error of `max(1e-6, abs(gold) * 1e-6)`. Failed calls count as incorrect in all rates.
The tolerance was selected before the full run completed. It does not repair
malformed JSON, choose different fields, reorder records or fill missing values.

Latency is the client HTTP round trip including response parsing and scoring.
Startup is excluded; initial connection establishment is included. Tractor uses
CPU inference over loopback, while the LLMs use remote cloud APIs. This measures
these deployments, including network overhead, rather than isolated model compute.
The CPU host is shared with an unrelated GPU workload. No GPU speed claim follows
from these measurements. P50 and P95 include failed calls; valid-response timings
are also reported. There is one run per provider; no P99 claim is made from 680 calls.

See [the report](../docs/benchmarks.md) for results and the scope of comparisons.
