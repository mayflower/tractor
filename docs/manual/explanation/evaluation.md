# Understanding accuracy and performance

The released tractor system is trained for a particular extraction contract. Its
synthetic acceptance results measure complete responses, including records, fields,
quantity conversions, and issue conventions. Field F1 measures extraction at a
finer level; it is not interchangeable with complete-response correctness.

On the frozen 680-request comparison, tractor has 94.41% exact complete responses,
compared with 24.85% for Haiku 5.5 and 54.71% for GPT-6 Luna. Median HTTP response
times are 330 ms on local CPU, 1,424 ms for Haiku's cloud API, and 1,873 ms for Luna's
cloud API. The [benchmark report](../../benchmarks.md) contains rounding-tolerant
scores, failure counts, hardware, model hashes, and the reproducible request set.

These are task-specific results. Tractor combines learned extraction with a
purpose-built normalizer; the LLM baselines generate their responses and arithmetic
from a shared prompt. Reasoning, tool use, provider structured-output modes, retries,
and answer repair were disabled. The comparison does not measure an optimized LLM
workflow or a general ranking of intelligence. Local-versus-cloud timings include
network overhead; GPU latency was not measured for the released bundle.

Synthetic texts are useful for controlled coverage of units, missing fields, and
record patterns. They do not establish accuracy on customer documents, OCR noise,
new domains, or different languages. Shared template families can make samples
correlated even when the literal texts are disjoint from training. The full 2,048
acceptance requests and the 680-request comparison subset are not independent tests.

A practical release gate combines a frozen acceptance set, failure review, and
end-to-end downstream checks. Measure the cost of wrong accepted values as well as
abstentions. A model extraction score is not a calibrated probability of correctness;
a valid JSON response is not evidence that the source was understood correctly.

For latency, choose the deployment you actually intend to use, wait for readiness,
separate startup from per-request work, and measure representative schemas and text
lengths. Report concurrent load, failure rate, median and tail latency together.
A bounded queue deliberately rejects excess work rather than accumulating it forever.

[Manual home](../index.md)
