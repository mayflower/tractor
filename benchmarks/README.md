# Benchmark the current release

Current pipeline Dev measurements are in [release/metrics.json](../release/metrics.json)
and explained in [the evaluation report](../docs/benchmarks.md). No matched LLM
comparison or latency measurement exists for model version 2026.10.21.

The 680-request `acceptance.jsonl` fixture is a narrow synthetic quantity/product
workload for comparison tooling. It does not represent the generalist Dev set or
establish an independent test of current training. Check source/template overlap
before treating any new run as held-out evaluation. `prompt.txt` supplies the LLM
contract; request outputs and credentials remain outside the source release.

For a new matched comparison, install with `nix develop` and `just setup`, start
the API, and use a new output directory. Cloud calls require provider credentials
in the environment and incur charges. Do not run GPU latency benchmarks on a
shared GPU without an explicit request.

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 tractor serve --model models/tractor --device cpu --port 8017
```

In a second terminal:

```sh
weights=$(jq -r '.weights_sha256' release/targets.json)
uv run --no-sync python -m tractor.compare --provider tractor   --data benchmarks/acceptance.jsonl --limit 680 --model-sha256 "$weights"   --url http://127.0.0.1:8017 --output reports/comparison-tractor
uv run --no-sync python -m tractor.compare --provider anthropic   --data benchmarks/acceptance.jsonl --limit 680 --output reports/comparison-haiku
uv run --no-sync python -m tractor.compare --provider openai   --data benchmarks/acceptance.jsonl --limit 680 --output reports/comparison-luna
uv run --no-sync python -m tractor.comparison_report   --tractor reports/comparison-tractor --haiku reports/comparison-haiku   --luna reports/comparison-luna --output reports/comparison-summary
```

These runners use prompt-only cloud baselines without retries or response repair.
Record the current hardware and exact provider/model revisions. Include failures,
reference coverage and local-versus-cloud network overhead in any interpretation.
The fixture runner's strict and numeric-tolerant comparison rules are separate
from the current typed pipeline's reviewed-equivalence Dev evaluation; their
numbers must not be interchanged.
