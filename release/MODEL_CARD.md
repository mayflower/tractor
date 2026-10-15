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
- system-one
---
# tractor · System One Extraction Model

![tractor — System One Extraction Model](../docs/brand/tractor.svg)

**Text in. Structured data out. Ready for your application.**

tractor is Mayflower’s **System One Extraction Model**, built for the repeated
extraction tasks inside catalog imports, supplier integrations and operational
software. Describe your fields; receive source-grounded text, separate product
records and quantities converted to the units your application needs.

One GLiNER encoder pass selects source spans. Deterministic Decimal arithmetic
normalizes quantities, and code assembles the JSON response. The local HTTP API
and lightweight Python SDK provide original offsets and explicit issues for
missing or ambiguous values. Run the model offline on your own CPU or GPU.

tractor is a specialized alternative to Haiku 5.5 and GPT-6 Luna for this extraction
contract. [Evaluation methodology and results](https://github.com/mayflower/tractor/blob/main/docs/benchmarks.md)
describe the measured workload and deployment conditions.

## Measured extraction performance

On 680 identical synthetic requests, the released weights achieve
**94.41% complete-response correctness** at **330 ms median CPU
HTTP response time**. Median responses are **4.3× faster than Haiku 5.5**
and **5.7× faster than GPT-6 Luna** in the measured deployments, with
higher complete-response correctness than both prompt-only baselines.

| System | Complete response correct | Data exact | Median | P95 | Failed outputs |
|---|---:|---:|---:|---:|---:|
| **tractor · CPU** | 642/680 (94.41%) | 94.41% | 330 ms | 520 ms | 0 |
| Haiku 5.5 · cloud API | 169/680 (24.85%) | 28.68% | 1,424 ms | 3,014 ms | 58 |
| GPT-6 Luna · cloud API | 372/680 (54.71%) | 59.41% | 1,873 ms | 3,456 ms | 10 |

![Extraction accuracy and median response times](../docs/brand/comparison.svg)

The comparison uses CPU loopback versus remote cloud APIs, disabled LLM reasoning,
no tools, retries or answer repair. It includes network overhead and covers a
specialized synthetic task, not optimized LLM pipelines or general capabilities.
[Results, tolerant scoring and reproducible evidence](https://github.com/mayflower/tractor/blob/main/docs/benchmarks.md).

## User manual

The [full user manual](https://github.com/mayflower/tractor/blob/main/docs/manual/index.md)
includes step-by-step tutorials, integration recipes, API/SDK reference, and design
explanations. Start with [your first extraction](https://github.com/mayflower/tractor/blob/main/docs/manual/tutorials/first-extraction.md),
then use the runnable Python, JavaScript, async, batch, database, and deployment examples.

## Use

In the [source checkout](https://github.com/mayflower/tractor), authenticate with
`hf auth login` using an account with access to this private repository:

```sh
nix develop
just setup
uv run --no-sync hf download mayflowergmbh/tractor --local-dir models/tractor
tractor serve --model models/tractor --device cuda
```

Use `--device cpu` for CPU inference. Serving loads local files offline. This bundle
requires the native tractor/GLiNER loader, not a generic Transformers pipeline.
See the [source README](https://github.com/mayflower/tractor#readme) and
[API documentation](https://github.com/mayflower/tractor/blob/main/docs/api.md)
for HTTP and SDK examples.

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

See [metrics.json](metrics.json) for integer counters. Real-world quality is unmeasured. CPU HTTP response times are measured above; GPU
latency is unmeasured for this bundle. Source/export equivalence
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

Source: [mayflower/tractor](https://github.com/mayflower/tractor).
Model: [mayflowergmbh/tractor](https://huggingface.co/mayflowergmbh/tractor). Both repositories are
private during preparation. The base model declares Apache-2.0; licensing for
this distribution is not yet selected. Preserve upstream attribution.
