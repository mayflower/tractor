# BiQwen branch evaluation

The BiQwen backend and decoder adaptation work through the existing Tractor API.
**The current weights are not a quality replacement for the released Tractor model.**
The modified record decoder removes the one-value-per-cell restriction, but lowers
record-field F1 slightly on this Dev corpus. Real contacts and career data remain below the 70% target.

## Full production-pipeline comparison

All **4,429 remaining Dev documents** were scored after permanent source-based
exclusion of CORD. No failed request from a permitted source was dropped. Values and issue codes were scored after the same programmatic normalization.
Numbers below are field F1, not raw span scores. The released Tractor checkpoint
was rerun in the new dependency environment: all 4,429 retained predictions match its
previous release evaluation.

| Evidence / output form | Documents | Tractor | BiQwen, modified decoder |
|---|---:|---:|---:|
| Single object, synthetic | 2570 | 99.09% | 95.26% |
| Entity list, synthetic | 320 | 99.40% | 96.08% |
| Repeated records, synthetic | 338 | 90.30% | 85.66% |
| Contracts, real / human labels | 54 | 71.03% | 74.77% |
| Document metadata, real / human labels | 143 | 85.71% | 85.93% |
| Advertising-order headers, VRDU / human labels; insufficient support | 4 | 100.00% | 50.00% |
| Contacts, real / deterministic labels | 200 | 79.15% | 66.16% |
| Careers, real / deterministic labels | 200 | 93.91% | 63.15% |
| Research, real / deterministic labels | 200 | 92.62% | 87.97% |
| Document metadata, real / deterministic labels | 200 | 97.29% | 96.81% |

Only four VRDU documents remain in the real invoice/order family; no general
invoice-domain conclusion is supported. CORD is absent from all splits, including
headers and derived windows.

The additional 200 real product documents have incomplete references; they were
evaluated, but neither model receives a product-domain aggregate from them.
Missing real evidence in other domains remains missing. Both models fail the
complete quality gate; synthetic scores cannot establish general reliability.

BiQwen completed all documents without runtime errors. Tractor rejected five
requests because their combined schema/text exceeded its 512-token context.
BiQwen permits 4,096 tokens. These failures remain in the reports.

## Isolated decoder comparison

All **338 Dev documents containing records** were evaluated using both decoders
on identical neural states from one backbone invocation. The modified predictions
match the full-run predictions for all 338 documents.

| Metric | Upstream single-value decoder | Modified multiple-evidence decoder |
|---|---:|---:|
| Record-field precision | 85.77% | 87.90% |
| Record-field recall | 86.65% | 83.54% |
| Record-field F1 | 86.21% | 85.66% |
| Completely correct requests, scored references | 41.70% | 32.10% |
| Field issue accuracy | 98.17% | 94.01% |

The request-exact denominator is 271; 67 requests have incomplete references.
The change preserves competing candidates and improves field precision, but loses
recall and complete-request accuracy. It is an experimental structural correction,
not a measured quality improvement. Both decoders produce the same anchors and
161 extra records, so anchor detection is another unresolved problem.

The next model change should train record assignment with multiple positive
occurrences per anchor and field, including equivalent repeats, conflicting values,
absent fields and neighboring records. Validate that training through this same
pipeline and calibrate separately. Decoder changes cannot recover values or
anchors that the proposal heads do not detect.

## Integration checks and observed failures

- **845 automated tests pass**, including exclusive occurrence ownership, retained
  multiple values, scalar ambiguity after normalization, capacity failures and
  rejection of altered model code.
- All 15 existing manual requests return HTTP 200; HTTP and SDK outputs agree.
  Only **3 of 15** match their quality expectations exactly. These product-heavy
  examples are diagnostics, not a representative quality benchmark.
- The real API converts `12,5 cm` to `125` mm. A contact example returns the person,
  email and normalized date correctly, but misses the telephone number.
- A two-person email example detects both people, but merges the first person’s
  two email addresses into an invalid span. Normalization rejects it with
  `invalid_email`. The second person’s email is correct.
- A short two-product example and a competing-width record example miss their
  record anchors. The normalization pipeline cannot repair missing records.
- Overlong input returns HTTP 422 `token_limit`; it is not truncated.
- The built Python wheel contains the backend and its checksum manifest.

## Reproducibility and limits

- BiQwen revision: `96184881677b03adbf28ed5ba18a17548f8fb861`.
- Tractor weights SHA-256:
  `f9b177aa18d1f968cf8e2b3a99cff8938c096be8570064182f84c5a16488b90a`.
- BiQwen backbone is BF16, heads FP32; Tractor is FP32. TF32 matrix multiplication
  is disabled for both. This compares these deployment configurations, not
  identical arithmetic across architectures.
- BiQwen source/anchor thresholds are the exported 0.5/0.5. Tractor uses its
  released 0.5/0.3 policy. These are distinct model policies; no thresholds were
  tuned on these results. Upstream confidence calibration does not calibrate the
  changed assignment rule.
- BiQwen was trained on a derivative of the Tractor corpus. Training overlap with
  this Dev corpus has **not** been excluded. This is a compatibility and quality
  comparison on Dev, not an independent generalization benchmark.
- Scores were recomputed from unchanged saved predictions after source exclusion.
  No training, model promotion, Hugging Face upload or latency benchmark was performed.

All domain/form aggregates, exact counts, runtime hashes and weight identities are
in [the machine-readable report](biqwen-evaluation.json).
See [setup, decoder rules and reproduction commands](manual/how-to/biqwen.md).
