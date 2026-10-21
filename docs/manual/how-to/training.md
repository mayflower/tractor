# Build and train a structured extraction model

The authoritative scope is [reqs.md](../../../reqs.md): copy explicitly stated
values into fixed fields, repeated records or typed entity lists. Classification,
routing, sentiment, urgency, QA, relation triplets, summaries and normalization
onto code systems are excluded. Grouping of explicitly described text records is allowed; table parsing and layout reconstruction are excluded. Training tools
support more task types internally; that does not make those types product goals.
Training uses the same typed `POST /v1/extract` contract as inference.

## Prepare realistic document data

Run commands inside `nix develop`. The document pipeline has separate planning,
authoring, review and native-target validation stages. Planning creates **no
training documents**. Count accepted examples only after validation and sample
review. The 30,000 figure is the first synthetic training target, not a statement
that those examples already exist.

```bash
uv run --no-sync python -m tractor.document_build prepare \
  --output data/my-documents --train-count 30000 --heldout-per-domain 200
uv run --no-sync python -m tractor.document_build render \
  --output data/my-documents --limit 64 --env ~/.env
uv run --no-sync python -m tractor.document_build validate \
  --output data/my-documents --model models/tractor
```

The 16 domain schemas live in `schemas/extraction`. Plans rotate fields, languages
and output forms. An offline Teacher first authors fictional facts, then writes
an appropriate document, then reviews its evidence, roles and realism in a
separate call. Fixed templates with substituted values are not accepted as the
new document corpus. Exact markers establish original offsets; they do not prove
factual consistency or realism. Same-model review remains Silver supervision.
Inspect a stratified sample before scaling and retain rejected examples/reasons.
A valid receipt or form can naturally use labels and tables.

Rendering is bounded and cached. Beyond 256 candidates, a passing `scale-audit.json`
must bind the frozen recipe to inspected document IDs covering every domain,
output form and language, with no unresolved findings. An automatic Teacher
acceptance flag is not this sample audit. Repeat it to process remaining planned cases;
failed candidates remain recorded and do not count toward quotas. Native validation
writes an immutable `validated` directory and checks the actual CPU collator,
context limits and record targets. Finish the intended batch before validating.
It is not a model quality evaluation and does not authorize training or promotion.
No automatic 30,000-case launch or quota-refill claim follows from a sample.

The shared durable reservation ledger is `reports/training/teacher-budget.json`.
[The budget](../../../configs/generalist-budget.json) enforces a cumulative USD 500
ceiling, including failed attempts. It is a conservative reservation, not a bill.
The pipeline loads only required provider variables from the specified env file.
Caches and private provenance stay outside publication artifacts.

## Import existing public extraction data

```bash
uv run --no-sync python scripts/fetch-document-sources.py
uv run --no-sync python scripts/fetch-document-sources.py --verify-only
uv run --no-sync python -m tractor.document_sources \
  --output data/my-public-documents --limit 200
uv run --no-sync python -m tractor.document_sources \
  --output data/my-public-documents --validate --model models/tractor
```

The source registry pins revisions and file hashes for Kleister NDA,
Kleister Charity and VRDU. DocILE remains conditional on its provider download
credentials. Code licensing does not establish data redistribution rights; resolve
registry restrictions before publishing data.

Adapters use supplied text/OCR as input, never images as an extraction task.
Literal fields are projected to original offsets. Normalized-only
Kleister labels are excluded and logged, not converted into invented spans or
negative examples. Official source splits remain separated. Text windows use
boundaries independent of gold answers. Unsupported record crossings and native
failures reject the document rather than quietly retaining an incomplete target.

Report original documents, text windows and sampled training exposures separately.
Only distinct source examples count toward the 200,000-real-example target. Long
reports may yield many empty windows; importing them does not establish a useful
training mixture. Do not mix this output blindly into training. Public sources
also need domain mapping, semantic sample audit, leakage checks and data-rights
review. Synthetic data never counts toward the public minimum.

## Assemble and qualify a training corpus

Current local build paths and counts belong in `reports/extraction-build`.

For full training, include every eligible German and English training episode
from the complete source pools and approved supplements. Do not replace the
training pool with a repair sample or per-domain quotas. Keep Dev, calibration
and test membership fixed. Repeated schedule exposures are not new documents.

`tractor.training_corpus` takes a frozen selection configuration with `languages`,
`sources`, `heldout` and `dev`. Source entries contain `path` and `sha256`.
Earlier sources take precedence for corrected annotations of the same episode;
conflicting text or task inventories fail. Every input row is recorded as included,
duplicate or excluded with a reason. Other configuration keys, including sampling
limits, are rejected.

`tractor.full_corpus_native` then compiles every selected episode through the public
API adapter and validates native spans, record bindings and token limits. Rejected
tasks remain in the audit. All Dev episodes remain in evaluation, including requests
that exceed model capacity. Successful compilation proves technical compatibility;
it does not constitute a new human semantic review of the corpus.

The local `configs/active-training.json` identifies the active dataset and training
configuration. A configuration's `tractor_dataset` binds its dataset path, manifest
SHA-256, unique episode count and native training request count. The trainer checks
these before loading a model, and rejects a substituted subset or modified files.
Bound full-dataset runs require complete epochs without step or wall-clock caps
or early stopping. Their data loader preserves the final partial batch.
Dataset preparation, a completed training run and a measured quality improvement
are separate results and must be reported separately.
A scope-filtered historical pool is not automatically realistic or balanced.
Retired deterministic coverage/classification configs must not be used to fill
new quotas. In particular, retained schematic synthetic rows need separate
exclusion from the realistic-document corpus.

For a targeted repair of repeated text records, `tractor_decoder_only: true`
freezes every parameter except `record_decoder` before optimizer creation. Set
`tractor_required_groups` to `["record_decoder"]` and disable LoRA. The encoder
runs without dropout; native record proposal supervision remains in training
mode. Training evidence verifies every frozen parameter, and the saved model
must still pass a full production-pipeline Dev comparison.
To train span recognition and record assignment together while preserving the
encoder, set `tractor_freeze_encoder: true` and require learning evidence for
`boundary_head` and `record_decoder`. This is mutually exclusive with decoder-only
training and LoRA. Encoder dropout stays disabled and all frozen encoder weights
are verified after training. For full fine-tuning, leave both freeze options false
and require evidence for the encoder and both extraction heads.
Use `eval_strategy: "epoch"` when the evaluation interval coincides with the last
partial gradient accumulation of an epoch: the pinned trainer does not execute
step-based evaluation on that trailing update. A run without any production
pipeline checkpoint evaluation is rejected.

An optional `tractor_regression_baseline` names the previous full-pipeline Dev
metrics file. Dev bytes, quality policy, thresholds and precision must match.
Checkpoint eligibility then limits supported field F1 drops to two percentage
points and domain/output F1 drops to half a percentage point. Supported rejection
scores and the total number of correct rejections may not decrease. These are
repair regression tolerances, not release thresholds or statistical confidence
claims. A run with no eligible checkpoint leaves the prior candidate active.

Use [the extraction policy](../../../configs/extraction-quality.json) for domain
and output-form gates: field F1 must exceed 70% in each declared group, with
record F1 also required for repeated records. `tractor.document_quality.evaluate`
calculates these gates from episode-aligned predictions. Its CLI writes an
immutable report and exits nonzero when a group fails:

```bash
uv run --no-sync python -m tractor.document_quality \
  --episodes data/my-qualified-corpus/dev.episodes.jsonl \
  --predictions reports/my-dev-baseline/predictions.jsonl \
  --output reports/my-domain-gates.json
```

 Missing real/human
assessment fails; synthetic results cannot substitute for it. The historical
family-level generalist callback is a diagnostic, not this domain qualification.
Product accuracy is a secondary regression check.

Record metrics count only `records` tasks. Header/entity tasks must not increase
record precision or recall. Checkpoint ranking uses `0.9 * minimum + 0.1 * macro
mean` over unique measured Dev gates, including public Silver fields. This allows
ranking when one group remains at zero; it does not relax the independent >70%
release gates or replace missing real evidence.

Record cells declared as single values can specify `cardinalities` using
`optional_one` or `required_one`. This enables scalar assignment and exclusive
span ownership. Preserve list cardinalities for genuinely multi-valued cells;
never discard gold spans to make a scalar schema fit. Run native target validation
when changing these contracts.

The public importer pins its VRDU split filenames and preserves all prior source
split memberships when expanding a split. Public near-duplicate quarantine needs
literal overlap: equal text after replacing every answer with a placeholder is
not evidence that two real receipts are the same document. Synthetic template
checks still use masked similarity.

Mixture qualification counts actual record tasks with at least two fields, not
just a document's `artifact` label. There is no mandatory public table/record source count. Only explicitly
described text records are eligible; VRDU position parsing is excluded.
Missing real evidence for in-scope text records remains explicit and cannot be
replaced with a table benchmark. A representative record window takes priority over a
header-only window from the same source. Public replay is capped at eight
exposures per source; counts and replay are reported separately. All prepared real
Dev cases enter selection (up to 200 per cell), alongside the fixed synthetic
sample. A source-count floor alone is not evidence of domain coverage.

Before training, freeze paired episode/native files, split membership, source
hashes and eligibility exclusions. Keep source documents and derived windows in one split.
All model selection uses Dev; calibration and test are held out. Verify encoder
and relevant extraction-head gradients and actual weight changes during a bounded
learning check. Never overwrite `models/tractor` or an existing run directory.
GPU training must respect shared GPU load. Do not run GPU latency benchmarks.

```bash
tractor train --model models/tractor --data data/my-qualified-corpus \
  --config configs/my-training.json --device cuda
```

## Measure development quality

```bash
tractor evaluate --model models/tractor \
  --episodes data/my-smoke/dev.episodes.jsonl --split dev \
  --device cpu --output reports/my-dev-baseline
```

The evaluator stores raw predictions and failures, full-field and element metrics,
one-to-one Record matches, complete requests and task-family macro F1. Extra
Records remain false positives. A model that abstains everywhere cannot obtain
high recall; precision with no predictions is undefined. Compare candidates on
identical source groups, with paired cluster intervals and a separate product-API
regression evaluation. Model selection happens on Dev only.

For general-model selection, prioritize task-family macro F1 and inspect real-text
and synthetic results separately. Treat product extraction as one regression suite,
with an explicit acceptance floor. A pilot design can disable relative regression
guards with `product_regression_metrics: []` and set
`minimum_product_metrics: {"exact_request_rate": 0.90}`. The report then selects
the qualifying candidate with the highest general-task macro F1. Record changes
to acceptance criteria; a revised business objective does not retroactively make
an earlier predeclared criterion pass.

## Continue training and calibrate

To warm-start, supply the chosen checkpoint to `tractor train --model`, choose a
new output directory and explicitly configure the new optimization budget.
Optimizer, scheduler and RNG state restart. Exact interrupted-run resume is not
supported by the pinned trainer; a weights-only restart must not be called resume.

[The conservative configuration](../../../configs/generalist-conservative.json)
provides a bounded 300-update warm-start recipe with separate encoder/task learning
rates. Supply your prepared dataset directory and change `output_dir` before each
run. Its replay mix must be constructed from training data only; the configuration
does not automatically mix or create data.

Freeze model and preprocessing before final calibration. The existing product
threshold path accepts `--split calibration --calibrate` with one frozen model;
it refuses checkpoint-directory ranking on this split. Native multi-task scores
remain raw: the product threshold cache must not be reused for them.
Do not use final test errors to generate the next training set.

For native threshold selection on a separate, completely annotated calibration
file, use:

```bash
tractor calibrate-native --model runs/my-training/final \
  --input data/my-training/calibration.episodes.jsonl \
  --output reports/my-native-calibration --device cpu
```

Source groups are split between score fitting and policy selection, with at least
ten groups in each half. A small monotone score mapping is measured using Brier
score and ECE on emitted fields; it is not substituted for raw inference scores.
Three native thresholds are evaluated with direct inference. The report separates
the normal policy and any policy meeting the observed 0.99 precision target.
That target is a point estimate, not a reliability guarantee. Missing Records and
unanswered fields remain visible through recall and coverage.

The resulting `native-policy.json` is tied to the exact weight hash and library
pin. To include it in an export, add its path as `native_policy` in the selected
evaluation JSON. The exported local extractor loads the policy; the existing
product API continues to use its separate Text/Quantity thresholds. Model changes
invalidate the native policy.

After freezing the bundle and policy, evaluate the reserved test file once:

```bash
tractor export --run runs/my-training --selection reports/my-selection.json \
  --output runs/my-candidate
tractor evaluate --model runs/my-candidate \
  --episodes data/my-training/test.episodes.jsonl --split test \
  --device cpu --output reports/my-final-test
just check-model runs/my-candidate cpu
```

The selection JSON must contain `model` (the checkpoint inside the supplied run),
`thresholds` (the separate product `text`, `quantity`, `anchor` policy), and optionally
`native_policy` (the calibration policy file). Export requires completed optimizer
steps and verified learning in every required module. A checkpoint directory alone
does not prove that a run finished. Keep the training evidence and raw evaluation
outputs outside the inference bundle. The HTTP smoke checks correctness; it is not
a latency benchmark.

Exports and publication are separate actions. Keep candidates local and repositories
private until quality, regression and export/reload checks justify promotion.

Table parsing and reconstruction of PDF/OCR columns are outside the product.
Run `python -m tractor.document_text_scope` on both qualified corpus inputs before
building a mixture. The filter preserves text, spans and split memberships, removes
public VRDU record tasks and quarantines structurally tabular synthetic record
tasks. Header fields remain eligible. Removed tasks are recorded in a sidecar audit.
A structural filter is conservative, not a semantic proof; ambiguous text records
require review. Generation and independent review must enforce the same boundary.

Numbered or unnumbered goods/quantity/code position blocks and inline position
lists are excluded too,
even without pipes, tabs or aligned columns. Apply the same scope filter to every
split, preserve eligible header tasks, and reevaluate both baseline and candidate
on the resulting Dev version before attributing a gain to training.

## Local field roles and bounded replay

Before building a mixture from public document windows, run
`python -m tractor.document_text_scope --roles --source INPUT --output NEW_OUTPUT`.
The source must already have passed its prior qualification. This additional gate
removes unverified absent fields from SWDE/Kleister/VRDU, requires explicit local
governing-law evidence for NDA jurisdictions and explicit total-value roles for
financial amounts, and retains VRDU header values only
when their label directly precedes them. It never guesses a replacement target.
Fields removed from a schema are unsupervised, not trained as absent. Authored
negative examples and original NER negatives remain available.

Kleister Charity windows with only organizational metadata belong to `metadata`,
not financial-metric coverage. Missing real financial evidence must stay missing.

The mixture builder requires this public role audit. Synthetic replay defaults to
at most three exposures per example per epoch (`--max-synthetic-repeats`). The
60/30/10 proportions are upper targets: when a cell has insufficient diversity,
the epoch shrinks and its deficit is reported in `unfilled_exposure_targets`.
Do not increase repetition merely to hide a shortage of distinct text examples.

## Evaluate the production pipeline

Document training defaults select with `eval_pipeline_selection_score`. Raw span
scores are diagnostics, not product gates. The document evaluator and HTTP API
execute the same extraction, evidence validation and normalization code. Final
fields are unique value lists; repeated records remain distinct. Expected values
come from independent annotations, not from applying the production parser to gold.

Corpus fields declare public types through an explicit source-schema policy.
This includes contact values, names, dates, amounts, quantities and identifiers,
as well as GB registered-office address components for Kleister Charity. Mixed
semantic fields and missing output parameters remain documented contract gaps.
Missing real evidence still blocks release. Changes to this output contract are
not model accuracy improvements and must not be compared directly with old span metrics.
Historical run configurations and reports retain their original meaning.

See [document processing](document-processing.md) for the production API contract.

## Migrate existing corpora to typed fields

Run the migration separately on qualified public and authored corpus roots:

```bash
nix develop -c uv run --no-sync python -m tractor.typed_corpus \
  --source QUALIFIED_CORPUS --output NEW_TYPED_CORPUS
```

The output directory must not exist. The migration checks source manifest hashes,
preserves every episode, split, source span, record and native training target, and
validates requests against the public API schema. A separate annotation converter
defines final targets without calling the production normalizers. The report and
per-field audit record type assignments, recovered targets and remaining gaps.
Existing independent final annotations are retained when their contract is unchanged.

The policy uses the source schema, not annotated answers or model predictions.
For example, a restaurant name and a player name become `organization` and `person`
respectively. An account identifier is not automatically an IBAN. A mixed
"candidate or employer" field stays `text`. A complete notice clause remains text
when reducing it to a duration would lose conditions such as quarter-end notice.
Missing currency, phone-country or target-unit parameters are never inferred from
gold values. Invalid synthetic phone numbers or IBAN checksums stay visible as
annotation gaps; the migration does not rewrite source evidence to make them valid.

Migrate an existing mixture directory with the same command, then attach the
migrated pools to its unchanged training exposure schedule:

```bash
nix develop -c uv run --no-sync python -m tractor.typed_mixture \
  --mixture NEW_TYPED_MIXTURE \
  --synthetic NEW_TYPED_AUTHORED_CORPUS --real NEW_TYPED_PUBLIC_CORPUS
```

This creates the corresponding typed training episode view and checks every
scheduled text against the existing native training row. Exposure order and
multiplicities are unchanged. Configure `tractor_extraction_dev` to a migrated
Dev episode file before the next training or evaluation run. Calibration and test
receive the same frozen policy; they do not select the policy or model recipe.
An inherited `training_ready` flag qualifies preserved native supervision only;
`normalization_ready: false` explicitly records incomplete final-target/contract
coverage. Missing final targets never count as successful extraction.

Persist `text_origin` and `label_origin` explicitly in every serialized episode.
Do not omit these fields when their values equal model defaults: legacy-origin
reconstruction can otherwise change synthetic text into apparently real text,
or deterministic labels into teacher labels. Verify the serialized/reloaded
episode, not just the in-memory migration result. `tractor.corpus_provenance`
can restore these two fields from frozen original identities; it rejects mismatched
text, source groups and splits and changes no requests or annotations.

## Train with final-value contracts and diagnose individual fields

Textual reference annotations must not impose requirements absent from the
request. A model designation with a trim suffix, or without a year that was
never required, can be a valid answer. Offline `value_equivalences` groups record
reviewed, source-supported alternatives for one field value. Any member, or
multiple members of the same group, satisfies that value. This affects evaluation
only; the API still returns its own normalized output. Numeric values, identifiers
and dates keep their type-specific comparison rules.

Use `reference_complete: false` when the reference cannot establish all acceptable
answers. The broad SWDE auto/camera model requests have this limitation by default:
their source cell annotations do not define permissible model-name boundaries or
an exhaustive set of page-level model mentions. A nonempty unreviewed alternative
is unresolved, not a demonstrated extraction error or a success. An empty answer
still misses known present evidence. Reports expose unresolved/reference counts;
incomplete-reference scores are conditional diagnostics and cannot select a
checkpoint or establish release quality. Fully unresolved requests are excluded
from the exact-request denominator. Do not compare the resulting diagnostic
fractions as though their reviewed subsets were identical.

Approve alternatives against the actual source and request, retaining review
provenance. Do not automatically accept substrings, similarly spelled names or
year-stripped values. Review decisions made after inspecting Dev predictions
correct the benchmark; they are not independent held-out evidence. Re-score all
compared checkpoints using the same versioned annotations and evaluation code.

Native training must compile the same schemas as the public API: anonymous field
IDs, record name and anchor, cardinalities, descriptions (including quantity
instructions), and one request per task. `tractor.api_training` converts frozen
typed episodes using `typed_requests_for` and `model.native_example`, then checks
native span offsets, record counts and every packed record's field bindings with
the training collator. Reusing older
task-named native files trains a different schema distribution. The converter
records token-limit exclusions and contradictory component supervision; it never
turns those into negative targets. All Dev episodes remain in pipeline scoring.

Declared source cardinalities survive typed migration. An implicit legacy list
default cannot replace an independently specified scalar; contradictory explicit
scalar contracts fail validation. Correcting an old request contract requires a
versioned corpus and reevaluation of both checkpoints on identical corrected
requests. That change alone is not evidence of a training improvement.

Scalar cardinality limits normalized values, not source mentions. Native decoding
therefore retains evidence lists so equivalent mentions can collapse and conflicting
values can produce an ambiguity issue. For scalar record fields, each selected
occurrence belongs to its highest-scoring record. Identical titles printed
separately beside two people remain separate occurrences. A single mention shared
across several people is not resolved by this mechanism; quarantine such training
targets instead of silently changing their meaning.

Checkpoint selection can declare `tractor_dev_thresholds` with `threshold` and
`record_threshold`. Use the same provisional settings for the starting checkpoint
and the training callback. This does not calibrate or modify a released bundle.
Both evaluators explicitly use the same FP32 matrix-multiplication settings and
restore the trainer's settings after evaluation. Reload and evaluate the saved
checkpoint before accepting a candidate.

`tractor_record_loss_weight` optionally sets the native GLiNER record objective's
positive, finite weight. It is persisted in the checkpoint configuration and
training evidence; inference thresholds are unchanged. Record assignment and
span losses have different reductions, so their scalar magnitudes alone do not
measure learning quality. Validate a changed weight on final record fields and
the other domains, retaining the same frozen Dev and baseline.

Episodes can declare `processing` and `locale`. A processed present answer must
carry an independently annotated `final_values` or `final_alternatives`, alongside
its original `items`/span alternatives. Numeric final annotations are canonical
Decimal strings. Native model supervision still uses source spans and record roles.
`final_annotation_missing` explicitly marks unavailable final targets; it never
counts as a successful field or complete request. Schema variants rename processing
keys with their fields. Production processing receives no final annotations.

Offline answers can also declare `expected_issues`, an exact set of API issue codes.
A present but invalid IBAN, for example, keeps its original span and declares
`final_values: []` with `expected_issues: ["invalid_iban"]`. The correct API result
is an empty value (`null` for a scalar or `[]` for a list) and that issue on the
same field and record. Returning `not_found`, returning the invalid value, or
emitting additional unrelated issues fails the outcome check. An absent field
expects `not_found`, or `missing_required_value` for a required cardinality.

Expected issues belong only to evaluation annotations, never to public request
fields. Rejection accuracy is reported separately and does not add true positives
to value-extraction F1. Field, record and complete-request accuracy compare both
values and issues. Diagnostics share the same type-aware equality rules as scoring,
and saved predictions retain Decimal precision. Missing annotations stay unscored;
an unsupported annotation grammar alone is not evidence of a required rejection.

`tractor.rejection_annotations` builds an independently annotated Dev view for
defined invalid/ambiguous outcomes. It never consults model predictions or invokes
the production normalizers. It preserves native span supervision and public
requests. Use `tractor.document_eval` for model scores and
`tractor.pipeline_attribution` for separate offline probes using annotated evidence;
probe results are diagnostic and never enter checkpoint predictions.

`tractor.pipeline_contract_data` adapts approved source contracts. A general movie
title request does not require a release year. Source-supported title forms with
and without a year may be equivalent when they identify one unambiguous film;
different dated films are not collapsed. Director lists use explicit newline
separation. UK registered-office roles are explicit.
Authored German numeric dates declare DMY; English numeric dates without an
explicit order stay unresolved. Original text and group/split membership are retained.
A separate, narrower annotation converter never calls production normalizers.

`tractor.reference_repairs` applies source-checked offline adjudications. Accepted
subtitles, honorifics, initial spacing and alternative printed designations belong
in explicit equivalence groups, not fuzzy string matching. Incorrect field-label
boundaries and merged people require corrected targets and native training spans.
Whole-address targets represent one address, not a list of its lines. Production
normalization can join adjacent selected address fragments when libpostal identifies
one address; it does not add unselected text or merge distinct address components.
Source-reviewed answers that convey the requested information must also count as
correct in the offline annotations. `apply_accepted_answers` records explicit
complete `final_alternatives`, bound to the reviewed source and field contract,
while preserving the original references, evidence spans and issue expectations.
Complete answer sets prevent accidental weakening: accepting `["Asian", "Asian,
Sushi"]` alongside `["Asian, Sushi"]` does not accept `["Asian"]` alone.
Keep these decisions outside inference and verify that the API requests and native
supervision are unchanged when reusing saved pipeline outputs.

Every compared checkpoint must use the same repaired references. Changed requests
or processing paths require fresh pipeline inference. Adjudicated Dev results
are not independent held-out evidence.

`tractor.pipeline_corpus` builds a bounded repair pilot: distinct text records,
metadata, broad retention, and up to three independent windows per charity source.
New supplement text must pass independent realism/semantic review, native capacity
checks and heldout-overlap checks. The full public reservoir remains available;
a repair pilot is not a replacement claim about dataset size or general reliability.

Reports include per-field support, precision/recall/F1, missing/wrong/extra values,
record matching diagnostics and representative failures. Field gates require
strictly over 70% F1 and at least 30 positive annotations by default; insufficient
or missing annotation remains an evidence gap. Checkpoint selection includes
supported field gates. Domain/shape gates and real-evidence requirements remain.
Do not compare selection scores across different field contracts or gate sets.

For a provisional Dev sensitivity experiment, `python -m tractor.document_eval`
accepts `--threshold`, for example `--threshold 0.3`. The report records the
actual threshold and whether it was explicitly overridden. This changes neither
the model bundle nor its calibrated deployment policy. Choose model recipes on
Dev, then use the separate calibration split for the final threshold; do not
publish Dev sensitivity results as independent test performance.

Entity fields and repeated records can be tested separately with
`--threshold 0.5 --record-threshold 0.3`. Both still use the production extractor
and one encoder invocation per text window. Always inspect per-field regressions:
a global lower threshold can improve record recall while adding false positives
to single-object fields. Deployment policies may declare an optional
`record_threshold` in `native-policy.json`; omitting it preserves the global
threshold. Any deployment thresholds require independent calibration for the
specified model hash, not copying a promising Dev setting.

## Controlled assignment-loss experiments

For decoder-only repairs, `tractor_assignment_positive_weight` enables a
training-only adapter for explicitly anchored records. Use `1.0` for the unchanged
native objective and diagnostics, then compare a separately trained candidate
with a bounded positive weight such as `3.0`. The adapter adds the positive BCE
term for matched list fields; scalar anchors, negative examples, object loss and
record matching retain the native objective. It never changes API inference or
normalization. Unsupported non-natural record groups fail explicitly.

Keep initial weights, data, seed, training steps, inference thresholds and Dev
identical across experiments. The training evidence records candidate-pair counts,
unweighted BCE gradient masses and the chosen weight. These are diagnostics;
only the full pipeline and unchanged regression guards determine acceptance.
Verify saved-checkpoint predictions against callback predictions.

Replay must include explicitly present ambiguous or invalid values as extraction
targets, so that normalizers can return their defined issues. A missing value is
a different target. Independently annotate final outcomes and mix these cases
with representative general extraction data. Synthetic documents must not exceed
three scheduled exposures per epoch. Never copy Dev failures into training.
