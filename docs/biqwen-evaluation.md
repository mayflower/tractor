# BiQwen branch evaluation

The BiQwen backend and decoder adaptation work through the existing Tractor API.
**The current weights are not a quality replacement for the released Tractor model.**
The modified record decoder removes the one-value-per-cell restriction, but lowers
record-field F1 slightly on this Dev corpus. Real contacts and career data remain below the 70% target.

## Full production-pipeline comparison

All **4,429 remaining Dev documents** were scored after permanent source-based
exclusion of CORD. After the quantity-description correction, all 315 affected
documents were rerun and the other 4,114 unchanged predictions retained. No failed
request from a permitted source was dropped. Values and issue codes were scored
after the same programmatic normalization.
Numbers below are field F1, not raw span scores. The released Tractor checkpoint
was rerun in the new dependency environment: all 4,429 retained predictions match its
previous release evaluation.

| Evidence / output form | Documents | Tractor | BiQwen, modified decoder |
|---|---:|---:|---:|
| Single object, synthetic | 2570 | 99.09% | 95.32% |
| Entity list, synthetic | 320 | 99.40% | 96.08% |
| Repeated records, synthetic | 338 | 90.30% | 85.68% |
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

Before the quantity-description correction, all **338 Dev documents containing records** were evaluated using both decoders
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

## What “the same training dataset” means here

The models share source material, but **not the same training selection**. The
pinned [extract-de-en dataset](https://huggingface.co/datasets/johannhartmann/extract-de-en/tree/53188d619b99b361c8a8c72b0d4df524d879f385)
contains 44,124 extraction training examples. Tractor's full run used 341,237
eligible DE/EN episodes (343,502 compiled requests). Native BiQwen deduplicates
texts and combines tasks, so these are document counts rather than equal numbers
of training requests.

| Source / domain | BiQwen extraction dataset | Tractor full training |
|---|---:|---:|
| Authored documents | 29,734 | 32,456 |
| Few-NERD | 6,000 | 123,785 |
| SWDE, all domains | 4,643 | 181,207 |
| SWDE contacts | 1,839 | 58,593 |
| SWDE careers | 638 | 26,154 |
| SWDE research | 557 | 40,672 |

The SWDE domain rows are subsets of its total. This reduced coverage is a plausible
contributor to the real-domain gap, not an isolated causal experiment.

An exact UTF-8 text-hash audit found **zero overlaps between the 4,429 Tractor Dev
documents and the published BiQwen extraction training split**. All 3,116 native
validation documents occur in Tractor Dev. Another 200 Dev texts occur in native
`test_human`; 1,113 have no exact match in the extraction dataset. The full comparison
is therefore development evidence, not a new independent benchmark. The audit does
not establish absence of near duplicates, pretrained exposure or overlap with the
separate decision corpus.

The [model export](https://huggingface.co/johannhartmann/biqwen-extract-decide-4b/tree/96184881677b03adbf28ed5ba18a17548f8fb861)
also records **83,740 combined training examples**, including decision tasks,
4,000 optimizer updates, batch size 2 and accumulation 8. Its model card identifies
update 3,800 as the selected checkpoint. That schedule permits at most 60,800
example presentations through selection, or 64,000 through update 4,000: less than
one pass over the combined corpus. The exported files do not contain the individual
training cursor or an example-coverage ledger. They cannot show which extraction
examples the selected checkpoint actually saw. This is a limitation of the supplied
training run, not a result of the Tractor adapter.

## Native-schema control experiment

Before the quantity-description correction, all **3,116 native validation documents** were run with their original declared
schemas, using unchanged weights, thresholds and arithmetic. Both original and
modified decoders received the same neural states. Their source spans went through
the ordinary Tractor normalization and issue handling; native labels never reached
inference. Native calibration and test rows were not run in this experiment.

For **2,509 single-task documents with identical schema text**, both native decoders
produce exactly the same final values and issues as the API. Opaque field ID changes
therefore do not explain the gap.

The other documents expose two input-format differences:

- In 292 single-task documents, the adapter adds GLiNER's quantity-description
  suffix, which was absent from native BiQwen training. Removing it changes 43
  outputs and improves field F1 on this subset from **95.01% to 95.52%**. Correct
  field outcomes increase from 1,480 to 1,493; correct expected rejections remain
  74/74. Thirty requests contain unscored references; request-exact accuracy uses
  the other 262 requests.
- In 315 documents, native input combines header and record tasks and includes a
  document-level table description. The public API sends separate declared
  requests. This changes 205 outputs, but restoring the original format does
  **not** restore quality:

| Input and decoder, same 315 record documents | Final field F1 | Completely correct scored requests |
|---|---:|---:|
| API schema, modified decoder | 86.29% | 32.67% |
| Native combined schema, modified decoder | 84.84% | 26.69% |
| Native combined schema, original decoder | 82.24% | 30.68% |

These field scores include header fields in the record documents. Request-exact
accuracy uses 251 requests with fully scored references; 64 have incomplete
references. The experiment changes task grouping and table descriptions together,
so it does not isolate either one's individual effect. It establishes that reverting
to the native input format and original decoder is not a quality fix.

### Adopted quantity-description correction

BiQwen now receives the caller's field descriptions verbatim. It no longer inherits
GLiNER's quantity-specific description suffix. Mandatory field types and the entire
normalization pipeline are unchanged. The fix was checked on **all 315 affected Dev
documents**, including records, and rescored together with all other Dev predictions.

| Affected synthetic domain | Before | After, final field F1 |
|---|---:|---:|
| Products | 91.74% | 92.24% |
| Logistics | 96.31% | 96.58% |

The existing per-field, domain, shape and rejection regression guard passes with
zero failures against the previous BiQwen API run. This is a small measured adapter
improvement, not a resolution of the real contacts/careers or record-assignment gaps.
The full release-quality gate remains failed. No weights or thresholds were changed.

## Integration checks and observed failures

- **851 automated tests pass**, including exclusive occurrence ownership, retained
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
- Exact text overlap with the published BiQwen extraction training split is zero;
  native validation overlap and other audit limitations are described above. This
  remains a compatibility and quality comparison on Dev, not an independent
  generalization benchmark.
- Scores were recomputed from saved predictions after source exclusion, then all
  quantity-affected documents were rerun for the adopted input-format correction.
  No training, model promotion, Hugging Face upload or latency benchmark was performed.

All domain/form aggregates, exact counts, runtime hashes and weight identities are
in [the machine-readable report](biqwen-evaluation.json).
See [setup, decoder rules and reproduction commands](manual/how-to/biqwen.md).
