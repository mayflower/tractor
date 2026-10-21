# tractor · System One Extraction Model

![tractor — System One Extraction Model](../docs/brand/tractor.svg)

**Text in. Typed, normalized data out.**

Version **2026.10.21** is a German/English generalist for explicit extraction:
names and contact details, identifiers, dates, monetary values, quantities,
document fields, typed entity lists and repeated records in readable prose.
A specialized GLiNER model finds the values. Python libraries parse, validate and
normalize them. One API returns `data` and `issues`: `POST /v1/extract`, also
available through `Client.extract`. Competing scalar candidates are returned
together, in source order, with `ambiguous_value`; missing values return null.

## Use

Download the complete private bundle with an authorized Hugging Face account:

```sh
nix develop
just setup
uv run --no-sync hf download mayflowergmbh/tractor --revision v2026.10.21 --local-dir models/tractor
uv run --no-sync tractor serve --model models/tractor --device cpu
```

Use the [tractor source](https://github.com/mayflower/tractor) and
[user manual](https://github.com/mayflower/tractor/blob/main/docs/manual/index.md).
This bundle uses tractor's native GLiNER loader, not a generic Transformers
pipeline. CPU and GPU serving are supported. Normalization runs in the Python
service; copy the complete bundle and use the pinned runtime dependencies.

## Current measurements

On 4,483 Dev documents, the complete pipeline achieves the following results on
the synthetic portion:

| Output form | Field F1 | Complete-record F1 |
|---|---:|---:|
| Single object | 99.09% | — |
| Typed entity list | 99.40% | — |
| Repeated records in prose | 90.30% | 57.61% |

Synthetic domain field F1 ranges from 93.19% to 99.57%. Real human-annotated
Dev field F1 is 71.03% for contracts (54 documents), 85.71% for metadata (143),
and 92.67% for invoices (58). Real texts with deterministic annotations are
reported separately in the [evaluation report](https://github.com/mayflower/tractor/blob/main/docs/benchmarks.md).
393 of 399 expected rejections are correct. Ten extra records are predicted.

These are checkpoint-selection Dev results, not held-out test accuracy.
Default bundle inference is checked against the selected checkpoint; verification
is recorded in `metrics.json`. No latency measurement or Haiku/Luna comparison
is claimed for these weights.

## Training

Complete training used 341,237 DE/EN episodes (16,149 German; 325,088 English),
compiled into 343,502 native requests. Four full epochs completed, totaling
42,940 optimizer updates and 1,374,008 sample exposures. The selected checkpoint
is epoch 3, step 32,205. The base is `fastino/gliner2.5-multi-v1` with pinned
GLiNER source; the source lock binds weights, evaluation pipeline and dataset.
The [dataset](https://huggingface.co/datasets/mayflowergmbh/tractor/tree/b07be20e73f60237ebb5a8aa37be4400a2d92097)
is private and includes provenance and exclusions.

## Known limits

This version is the chosen generalist default, with explicit acceptance of the
reported tradeoffs. Strict release qualification remains false: 12 regression
checks fail, complete-record F1 is below 70%, and real evidence is missing for
many domains. Restaurant cuisine, real product prices, company names and real
contracts need further improvement. Short label/value snippets and several
equivalent field descriptions can fail; see the evaluation report for examples. Incomplete references leave 604 field
observations unscored. Missing evidence is never reported as success.

The joint input/schema budget is 512 tokens. No table/layout parsing, OCR,
classification, routing, free-form QA, generic relation extraction or reasoning
about unstated facts is supported. Fields must declare their type and required
normalization context. Thresholds are provisional; calibration and test are held
out. No claim of uniform reliability follows from the synthetic scores.

## Provenance and distribution

Weights SHA-256: `f9b177aa18d1f968cf8e2b3a99cff8938c096be8570064182f84c5a16488b90a`.

Both GitHub and Hugging Face remain private. Release licensing has not yet been
selected; preserve upstream attribution. See `source-lock.json` and `metrics.json`
for the exact training, evaluation and publication scope.
