# Extraction benchmarks

**Typed extraction, measured after normalization.** These results describe tractor
model version **2026.10.21**, the selected checkpoint from complete German/English
training. Every score measures the production API pipeline, including parsing,
normalization, field assignment and issue handling.

## Evaluation contract

Evaluation uses 4,483 frozen Dev documents: 3,228 synthetic documents with teacher
annotations, 1,000 real documents with deterministic annotations, and 255 real
documents with human annotations. A document can contain several extraction tasks.
This Dev set selected the checkpoint. It is not an independent held-out test.
Calibration and test remain unused; thresholds 0.5 for fields and 0.3 for records
are provisional. No latency measurement or Haiku/Luna comparison has been run
for these weights.

Fields accept reviewed, source-supported equivalent answers after normalization.
Incomplete references leave unresolved answers unscored. They do not count as
correct. Missing values and expected rejection issues are evaluated explicitly.
Field F1 measures individual fields; complete-record F1 requires the whole record
to be correct. Neither is a complete-request accuracy claim.

## Synthetic documents by output form

| Output form | Field F1 | Complete-record F1 |
|---|---:|---:|
| Single object | 99.09% | — |
| Typed entity list | 99.40% | — |
| Repeated records in prose | 90.30% | 57.61% |

The repeated-record result remains below the 70% target. These are explicitly
described records in prose, not table parsing or layout reconstruction.

## Synthetic documents by domain

Each domain contains approximately 200 Dev documents. Synthetic coverage does
not establish reliability on unseen customer documents.

| Domain | Field F1 |
|---|---:|
| Invoices and receipts | 98.20% |
| Contracts | 98.97% |
| Forms and applications | 96.55% |
| Banking | 99.57% |
| Communication | 98.92% |
| Companies and contacts | 95.25% |
| Insurance | 98.93% |
| Clinical documents | 98.89% |
| CVs and job postings | 95.38% |
| Product attributes | 98.37% |
| Document metadata | 96.68% |
| Financial reports | 98.54% |
| Scientific literature | 93.19% |
| News | 97.27% |
| Logistics | 99.31% |
| Technical reports | 97.18% |

## Real documents

Deterministic annotations are automatically derived references on real source
texts. They are reported separately from human labels.

| Domain | Reference labels | Requests | Field F1 |
|---|---|---:|---:|
| Scientific literature | Deterministic | 200 | 92.62% |
| CVs and job postings | Deterministic | 200 | 93.91% |
| Document metadata | Deterministic | 200 | 97.29% |
| Companies and contacts | Deterministic | 200 | 79.15% |
| Contracts | Human | 54 | 71.03% |
| Document metadata | Human | 143 | 85.71% |
| Invoices and receipts | Human | 58 | 92.67% |

Real product references are incomplete, so no aggregate product F1 is claimed.
Human-annotated evidence is missing for most domains and for repeated records.
Counts are modest, especially for contracts and invoices.

## Issues and remaining limitations

Competing scalar candidates are returned together with `ambiguous_value`. Field
precision includes these returned candidates; compare models using this same API
policy. The same source can therefore expose several values for a scalar field.

Of 399 expected rejections, 393 are correct (98.50%). Ten extra records are
predicted across Dev. Five requests exceed the supported runtime capacity and
remain in the evaluation. There are 604 unscored field observations; aggregate
complete-request scores would have different denominators as answers change,
so they are not presented as a universal accuracy number.

The current version was explicitly selected for its broader generalist behavior.
The strict quality gate and 12 regression checks remain failed; publication does
not turn those failures into passes. Remaining weaknesses include full record
completion, restaurant cuisine (60.00% field F1), real product prices (70.59%),
company names and real contracts. Real product prices are an individually scored
field; the incomplete product-domain reference set still prevents an aggregate.
Short label/value snippets can also have incorrect span boundaries. For example,
`Breite: 12,5 cm.` currently yields `unsupported_number_format`, while the full
sentence in the tutorial extracts 125 mm. Repeating the same requested quantity
under several equivalent field names can return `not_found`. These observed smoke
failures remain known model limitations. Some additional fields have smaller measured regressions. Full checks and
support counts are in [metrics.json](../release/metrics.json).

## Training and reproducibility

The training corpus contains 341,237 eligible source episodes: 16,149 German and
325,088 English, compiled into 343,502 native training requests. All requests were
used in each of four complete epochs: 42,940 optimizer updates and 1,374,008 sample
exposures. Checkpoint selection chose epoch 3, update 32,205. Train/Dev/calibration/
test membership stays separate; no sampling or per-domain quotas replaced the
full training corpus. Technical validation is not an exhaustive human semantic audit.

The [private dataset](https://huggingface.co/datasets/mayflowergmbh/tractor/tree/b07be20e73f60237ebb5a8aa37be4400a2d92097)
contains the native requests, provenance, identities and exclusion audit. The
[release targets](../release/targets.json) pin the published model revision and
weight checksum. The bundle source lock records the pipeline implementation,
dataset revision, training exposure and selection provenance.

Specialized extraction combines a model that selects explicit values with
programmatic normalization. That focused approach is intended to reduce overhead
compared with a general-purpose LLM. Numerical speed or quality comparisons with
Haiku or Luna require a fresh matched benchmark of this exact bundle; none is
claimed here. See the [benchmark tooling](../benchmarks/README.md).
