# Extraction benchmarks

**The released tractor model returns more completely correct responses and responds
faster than the prompt-only Haiku 5.5 and GPT-6 Luna baselines on this specialized
synthetic extraction workload.**

On the same 680 requests, tractor achieves **94.41% complete-response
accuracy** with **330 ms median CPU response time**. Its measured median
is **4.3× faster than Haiku 5.5** and **5.7× faster than
GPT-6 Luna** in these deployments. These ratios include remote API network overhead.

![Accuracy and median response times for the frozen extraction benchmark](brand/comparison.svg)

## Same requests, measured results

| System | Complete response correct | Data exact | Median | P95 | Failed outputs |
|---|---:|---:|---:|---:|---:|
| **tractor · CPU** | 642/680 (94.41%) | 94.41% | 330 ms | 520 ms | 0 |
| Haiku 5.5 · cloud API | 169/680 (24.85%) | 28.68% | 1,424 ms | 3,014 ms | 58 |
| GPT-6 Luna · cloud API | 372/680 (54.71%) | 59.41% | 1,873 ms | 3,456 ms | 10 |

Measured on 2026-10-15. Each provider received all 680 frozen requests in the same
order, with one outstanding request at a time. Tractor ran in FP32 on an AMD Ryzen
Threadripper PRO 3975WX with two CPU threads, over local HTTP. Haiku and Luna ran
through their remote APIs. The CPU host was shared with an unrelated GPU workload;
tractor did not use the GPU. No GPU latency or isolated-compute speedup is claimed.

Complete-response correctness requires every field, record, number and issue code
in the expected order. This strict score includes decimal precision and issue
conventions specific to tractor; the data-only and tolerant scores help separate
those requirements from extraction quality. Data-only correctness ignores issues.
Failed outputs include
HTTP/transport failures, malformed JSON and responses rejected by the runner's
response contract. They count as incorrect and remain in latency percentiles.

## Does decimal rounding explain the difference?

The additional numeric-tolerant scores allow an absolute error of `1e-6` or a
relative error of `1e-6` against the gold value, whichever is larger. Field names,
strings, record order, missing values and structure must still match.

| System | Complete response, numeric tolerance | Data only, numeric tolerance |
|---|---:|---:|
| tractor | 94.41% | 94.41% |
| Haiku 5.5 | 25.00% | 28.82% |
| GPT-6 Luna | 56.18% | 60.88% |

## What the comparison establishes

Specialization is the reason to build a System One Extraction Model. Tractor is
trained for this extraction contract, selects source spans in one encoder pass,
and uses deterministic code for quantity conversion and response construction.
We expect that focused system to outperform a general-purpose LLM on its supported
tasks. The measurements above quantify the advantage for this particular workload.

This is a comparison of complete extraction systems: learned span selection plus
code in tractor, versus prompted text generation and arithmetic in the LLMs. It
is not an architecture-controlled experiment or a general intelligence ranking.
Haiku and Luna use the same published contract prompt with reasoning disabled,
4,096 maximum output tokens, no tools, no retries and no response repair. Provider
structured-output modes and provider-specific prompt optimization were not tested.
Those techniques could improve the LLM results, with their own latency and cost
tradeoffs. These numbers do not establish superiority over optimized LLM pipelines.

The inputs are 680 distinct synthetic source texts, selected before inference from
the release acceptance set using stratified hash ordering. They cover 38 scenario
labels, 341 German requests and 339 English requests. The model was trained for this
task distribution; source texts are disjoint from training/dev but template families
can be shared. They are not customer documents. A single run per provider measures
the stated deployments, including cloud/network variability and initial connection
establishment. Model startup is excluded; no explicit warmups were performed.

## Evidence and reproduction

- [Frozen requests and expected responses](../benchmarks/acceptance.jsonl)
- [The exact LLM prompt](../benchmarks/prompt.txt)
- [Machine-readable results, per-scenario counts, usage and provenance](../benchmarks/results/summary.json)
- [Run and rescore instructions](../benchmarks/README.md)

The evidence includes sanitized per-request responses and timings. Manifests record
the data, prompt, runner and released-weight hashes. Confidence intervals in the
JSON report use the Wilson method; shared synthetic templates limit an independent-
sample interpretation. P99 is deliberately omitted for this 680-request sample.

The main cloud runs used approximately **$0.2079** in API tokens
(Haiku **$0.1274**, Luna
**$0.0805**), estimated from returned usage at the
published [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) and
[OpenAI](https://developers.openai.com/api/docs/pricing) standard rates on the run
date. One failed Luna call returned no token usage. This is not an invoice and
excludes connectivity pilots. Local CPU hosting,
electricity and operations are not costed; tractor is not presented as free compute.

## Broader release evaluation

These measurements apply to the same distributed weights, identified by SHA-256
in [metrics.json](../release/metrics.json). The 680-request comparison is a subset
of the acceptance evaluation below, not an additional independent holdout.

| Dataset | Requests | Exact responses | Field F1 |
|---|---:|---:|---:|
| Synthetic acceptance | 2,048 | 1,898 (92.68%) | 0.9876 |
| Synthetic regression | 20,000 | 19,610 (98.05%) | 0.9974 |
| Authored API regressions | 32 | 32 (100%) | 1.0000 |

Acceptance covers product names, incomplete records, wide schemas, competing values,
repeated equal values and general quantities. Regression has 18,000 unique source
texts; some requests share text but vary schemas or units. Authored fixtures are
known regression targets. Checkpoint and threshold selection used dev data only.
The broader native evaluator uses a very small numeric tolerance (`1e-32`) while
the HTTP comparison above uses exact JSON-number equality for its primary score.

These datasets differ in difficulty and are not interchangeable accuracy estimates.
Real-world accuracy, OCR, unrestricted tables and other languages are unmeasured.
The full acceptance set has 150 incorrect complete responses, chiefly record and
wide-schema cases. The joint text/schema budget is 512 tokens.
