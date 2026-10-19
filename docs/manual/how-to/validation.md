# How to validate an integration and diagnose failures

## Check the environment and API contract

From the checkout inside the Nix shell:

```sh
just check
just check-model models/tractor cpu
```

The first command checks the lock, lint, fixture contracts, manual links/snippet
syntax, and unit tests. The second uses the actual model and checks real HTTP behavior. These are correctness
checks, not a latency benchmark. To run the manual examples against your server:

```sh
uv run --no-sync python examples/manual/python_client.py
uv run --no-sync python examples/manual/async_http.py
uv run --no-sync python examples/manual/check_examples.py
```

The final command runs the published request cases and verifies expected data,
issues, and normalized values. It does not call paid cloud APIs.

## Build a task-specific acceptance set

Collect representative requests and independently reviewed expected outputs. Include
missing fields, zero values, conflicting measurements, repeated names, multiple
records, Unicode, both supported languages, and the locales you actually ingest.
Test wide schemas and long record lists explicitly. Keep development examples for
schema/prompt/threshold choices separate from final evaluation examples.

Measure complete-record correctness and the rate of confidently wrong values,
not just JSON validity. Track abstentions, incorrect associations, and downstream
review volume. A no-issue response can still be wrong. See
[the release benchmark](../../benchmarks.md) for measured synthetic results and
[the frozen comparison](../../../benchmarks/README.md) for reproducible scoring.

## Troubleshooting

| Symptom | Check and action |
|---|---|
| Private Hub repository appears missing | Log in with `uv run --no-sync hf auth login` and verify account access. |
| Model file missing | Download the full repository into the exact `--model` path. |
| Native libraries fail to import | Enter `nix develop`, run `just setup`, and keep using `uv run --no-sync`. |
| SDK-only setup cannot start serving | `just setup-sdk` excludes the ML stack; use `just setup` in the server checkout. |
| Startup stays unready | Inspect the process log for missing files, incompatible normalization policy, or unsupported precision/device. |
| 422 `invalid_request` | Check key spelling, target units, field limits, locale, and anchor/many requirements. |
| 422 `token_limit` | Reduce the text or schema; the budget covers both, not text alone. |
| 422 `output_capacity` | Split crowded records/fields and keep associations intact. |
| 503 `overloaded` | Reduce concurrent calls or use a bounded queue and backoff. |
| 500 then 503 `not_ready` | Investigate the inference failure and restart the process. |
| Width is null | Inspect `issues`; set the correct locale and provide complete source units. |
| Two products become one desired entity | Anchors represent occurrences; entity resolution belongs to the application. |
| Source highlights move after emoji | Use Unicode code-point offsets, not UTF-16 or UTF-8 positions. |
| Decimal serialization raises TypeError | Use `tractor.jsonio.dumps`, not a float conversion. |

To measure your own serving latency, use `tractor bench --help` and an otherwise
idle device. Do not compare a busy shared GPU against the published CPU timings.
The manual's examples are integration checks, not new evidence of general accuracy.

[Manual home](../index.md)
