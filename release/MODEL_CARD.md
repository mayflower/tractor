---
language:
- de
- en
base_model: fastino/gliner2.5-multi-v1
tags:
- gliner2
- information-extraction
- quantity-normalization
- tractor
---
# tractor · System One Extraction Model

tractor extracts structured fields and scalar quantities from German and English
text. A single GLiNER encoder pass selects original source spans; deterministic
Decimal arithmetic converts quantities to requested units. The HTTP API and Python
SDK support document fields, anchored records, explicit issues and source offsets.

## Use

Place the bundle in `models/tractor` in the source checkout:

```sh
nix develop
just setup
tractor serve --model models/tractor --device cuda
```

Use `--device cpu` for CPU inference. Serving loads local files offline. This bundle
requires the native tractor/GLiNER loader, not a generic Transformers pipeline.
See the source README and `docs/api.md` for HTTP and SDK examples.

## Evaluation

| Dataset | Requests | Exact responses | Field F1 |
|---|---:|---:|---:|
| Synthetic acceptance | 2,048 | 92.68% | 0.9876 |
| Synthetic regression | 20,000 | 98.05% | 0.9974 |
| Authored API regressions | 32 | 32/32 | 1.0000 |

Exact responses include all records, fields and issue codes. Acceptance source
texts are disjoint from training/dev, with some template families shared with other
evaluation sets. Regression requests contain 18,000 unique texts; schemas and units
can vary for the same text. Authored cases are known regression targets. Selection
used dev data only. The datasets differ in distribution and difficulty.

See `metrics.json` for integer counters. Real-world quality is unmeasured. No
validated latency figure is published for this bundle. Source/export equivalence
and real CPU HTTP, SDK, Unicode offsets, token rejection and one-encoder checks pass.

## Limits

Long record lists, wide schemas and competing values remain fallible. There are
150 incorrect complete responses in the acceptance set and 390 in the regression
set. OCR, arbitrary tables, other languages and unrestricted documents are not
validated. The joint schema/text limit is 512 tokens. Missing or ambiguous values
produce null plus a structured issue; unsupported quantities are not guessed.

## Technical provenance

Base model: `fastino/gliner2.5-multi-v1`, revision
`cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3`.
GLiNER2 commit: `b86d35c02f2c1a4d9840ea50197bd1b6d13b031e`.
Training uses synthetic German/English data with occurrence-specific spans and
independent arithmetic gold. The exported weights use FP32; training used BF16
activations. Thresholds: text=0.25, quantity=0.25, anchor=0.5.

Weights SHA-256: `209a62ad03f559d7fd21fca9e44c4daad597825b1b6520df72545823bbc9c61b`.

Source: `mayflower/tractor`. Model: `mayflowergmbh/tractor`. Both repositories are
private during preparation. The base model declares Apache-2.0; licensing for
this distribution is not yet selected. Preserve upstream attribution.
