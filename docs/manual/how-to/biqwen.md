# Use BiQwen behind the Tractor API

This experimental branch supports
[johannhartmann/biqwen-extract-decide-4b](https://huggingface.co/johannhartmann/biqwen-extract-decide-4b)
at revision `96184881677b03adbf28ed5ba18a17548f8fb861`. The local model directory
selects the backend. Requests, the Python SDK, field types, normalization and the
`data`/`issues` response remain the same.
Field descriptions are passed verbatim, matching BiQwen's training format;
GLiNER-specific quantity-description additions are not applied.

## Install and start

Access to the model repository must already be configured through Hugging Face.
The download needs approximately 8.5 GB of disk space. Runtime code, tokenizer,
configuration and both weight files are checked against the bundled SHA-256 lock
before the model code is imported. The server loads local files only.

```sh
nix develop -c just prepare-biqwen
nix develop -c just serve-biqwen
```

The second command starts the API on `http://127.0.0.1:8002`, using CUDA and a
BF16 backbone with FP32 extraction heads. It uses the upstream PyTorch GatedDeltaNet
implementation and pinned Transformers 5.20.1. No FLA installation or runtime
package installation is needed. It performs one backbone invocation per request
and generates no tokens. To use FP32, pass `--precision fp32` to `tractor serve`.

```python
from tractor import Client, Quantity

with Client("http://127.0.0.1:8002", timeout=120) as client:
    result = client.extract(
        "Die Breite beträgt 12,5 cm.",
        {"width_mm": Quantity("Breite", unit="mm")},
        locale="de-DE",
    )
    print(result.data, result.issues)
```

The model supplies source spans; Tractor converts this example to `125` mm.
BiQwen's classification, decision and reference-resolution features are disabled
in this backend. The only extraction endpoint remains `POST /v1/extract`.

## Repeated records and competing values

The supplied BiQwen record decoder selects at most one span per cell. That loses
list values and prevents Tractor from detecting conflicting scalar values.
This branch replaces that decoder **at inference time**, without changing weights:

1. Accept source candidates using the checkpoint's source threshold (0.5).
2. Compare each candidate's assignment score with each record's learned ABSENT score.
3. Assign each accepted occurrence to the record with the greatest positive margin.
   An exact ownership tie fails explicitly with `ambiguous_record`.
4. Retain all accepted occurrences for each field in that record, including scalar
   fields. Repeated occurrences remain separate internally.
5. Run the ordinary Tractor pipeline. Equivalent normalized values collapse;
   competing scalar values return an array plus `ambiguous_value`. A requested
   list retains distinct normalized values.

This decision rule is `biqwen-multiple-evidence-v1`. The assignment head was trained
for single-value cells. The new margins are **not calibrated multi-value
probabilities**, and the original model card's scores do not describe this decoder.
The change removes a structural restriction; it does not establish that the
weights reliably recognize every value or assign it to the right record.
Confidence calibration from the upstream export is not reused as evidence for
the changed decoder. No confidence fields are exposed in the public API.

The model retains its 4,096-token joint document/schema limit, 48-token maximum
span length, 48 candidates per field and 16 records. Input overflow and detected
output-capacity exhaustion produce HTTP 422. Long spans outside the proposal
window may still be missed; this is a model limitation, not a parser fallback.

## Reproduce the comparison

Use the full Dev corpus and the same post-normalization evaluator as Tractor:

```sh
nix develop -c uv run --no-sync python -m tractor.document_eval \
  --model models/biqwen-extract-decide-4b \
  --episodes data/full-training-de-en-native/dev.episodes.jsonl \
  --output reports/biqwen-dev-new \
  --device cuda --precision bf16
```

The output directory must not exist. The evaluator records every prediction or
error, weights, runtime sources and precision. Record failures remain in the
evaluation. GLiNER threshold overrides do not apply to BiQwen. This is a Dev
compatibility/quality comparison, not final calibration or an independent test.
The exact-text audit found no overlap with the published extraction training
split, but all native validation texts overlap Tractor Dev. Near duplicates,
pretraining and the separate decision corpus are outside that audit. No latency
measurement is part of this run.

Results are recorded in [the branch evaluation](../../biqwen-evaluation.md).

To isolate the decoder change, compare the original and modified decoders on all
Dev episodes containing records. Both receive the same neural states from a
single backbone call; both are scored after the same normalization:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nix develop -c uv run --no-sync \
  python scripts/compare-biqwen-decoders.py \
  --episodes data/full-training-de-en-native/dev.episodes.jsonl \
  --output reports/biqwen-decoder-comparison-new
```

## Diagnose data selection and schema differences

With the pinned `extract-de-en` JSONL files under
`data/biqwen-source-audit/data`, reproduce the read-only overlap and schema audit:

```sh
nix develop -c uv run --no-sync python scripts/audit-biqwen-data.py \
  --output reports/biqwen-dataset-audit-new
```

This compares exact text hashes across splits, source counts and the input schema.
It records file hashes and never runs inference on calibration or test data.
It audits the current checkout's schema compiler. The report's historical 292
quantity-description differences refer to the compiler before the adopted fix.

To isolate the original input format, evaluate all 3,116 native validation documents
using their declared schemas, both decoders and the ordinary Tractor normalizers:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nix develop -c uv run --no-sync \
  python scripts/compare-biqwen-schemas.py \
  --output reports/biqwen-native-schema-new
```

This is an internal diagnostic, not a second extraction API. The script rejects
non-validation native rows, mismatched task bindings and reference resolution.
Only text and the declared schema reach the model; native gold annotations do not.
The default comparison uses the saved CORD-free API predictions from before the
quantity-description fix, reproducing the control experiment. Use `--baseline`
to provide a fresh full-pipeline run if the serving implementation changes.

## Full training

The next run uses `configs/biqwen-full.json` and `data/biqwen-full-de-en-v2`.
It continues the pinned merged BiQwen weights with LoRA plus extraction heads.
Decision and reference-resolution heads are frozen and have no training objectives.
The supplied notebook trainer is not used.

The compiler consumed **all 341,429 eligible DE/EN source documents**, before the
GLiNER-specific context exclusions. It produced **342,147 training requests from
339,949 documents**: 15,940 German and 324,009 English. There are 337,410 entity/object
requests and 4,737 record requests. No source quotas, sampling or product replay
are applied. Schemas, descriptions and per-task request boundaries are identical
to the public API. Multiple source occurrences remain separate training targets;
record assignment learns every positive occurrence against the ABSENT score.

The corrected recipe encodes each field's simple type alongside its description.
Declared processing parameters are distinguished from source selection: requesting
millimetres still permits extracting a source value expressed in centimetres.
The record scorer adds a jointly anchor/field-conditioned nonlinear residual,
initialized to preserve the old scores. Both candidate and ABSENT logits use this
same condition. Base bundles retain their legacy input encoding; new checkpoints
explicitly bind `biqwen-types-v2` and `biqwen-joint-record-v2`.

The internal encoding preserves the description and adds `Type: <simple type>`.
Parameter handling is fixed across all types:

| Parameter | Internal meaning |
|---|---|
| Address `component` | Declared source selection |
| Identifier `format` | Validation/output standard; `isbn13` also permits an ISBN-10 source for conversion |
| Quantity `unit`, money `currency` | Processing/output contract; source spelling and units are retained |
| `country`, field/request `locale`, `date_order` | Explicit processing context; missing context is never supplied |
| `allow_partial`, `include_adjacent_year`, `prefixes`, identifier `uppercase`/`remove_spaces`, list `separator` | Declared processing options; non-default values are encoded |
| `cardinality` | Final API contract; does not suppress competing source occurrences |
| Address `group` | Internal assembly binding, not neural source-selection text |

Types without parameters receive their type and description only. Non-null
request locale is used when there is no explicit field locale. Optional defaults
are omitted to avoid unnecessarily consuming the joint 4,096-token context.

Head loss weights are explicit in the recipe. The causal Qwen backbone and the
two-layer bidirectional adapter are retained; increasing their capacity is not
assumed to improve quality. The [implementation checklist](../../../abhakliste.md)
records outstanding work and the evidence required before full training.

Every source task is accounted for. Of 343,694 training tasks, 1,547 are quarantined:

| Technical exclusion | Tasks |
|---|---:|
| Source span splits a Qwen tokenizer token | 1,228 |
| Source span exceeds the model's 48-token proposal width | 316 |
| Too many source occurrences for a field's candidate capacity | 3 |

The compiler never widens annotations, drops only the difficult field, or converts
an unrepresentable present value into an absent target. The per-task ledger is
`data/biqwen-full-de-en-v2/rejections.jsonl`. This is complete technical validation;
it is not a new exhaustive human semantic review of all source annotations.

All **4,429 Dev documents** remain in the final pipeline evaluation, including
requests that cannot be compiled into native training targets. Calibration and
test IDs, groups and normalized text hashes are excluded from training and never
used to select a checkpoint. CORD remains excluded from every split.

Verify the frozen data, runtime, compiler, regression baseline and update schedule:

```sh
nix develop -c just check-biqwen-full
```

This performs no model inference or optimization. The planned run has four complete
epochs, batch size 2 and gradient accumulation 8: **21,385 updates per epoch,
85,540 updates and 1,368,588 request presentations in total**. The final partial
batch is retained. There is no maximum-step cutoff, Dev subset or early stopping.

The real-model memory/gradient check is a separate operation without optimizer
updates or latency measurements. Use a fresh report directory:

```sh
nix develop -c uv run --no-sync python -m tractor.biqwen_preflight \
  --config configs/biqwen-full.json --output reports/biqwen-gpu-preflight
```

It covers short DE/EN requests, multiple records, dense evidence, the full input
length and 32 fields. It checks that weights remain unchanged. Passing this check
establishes technical feasibility for those shapes, not extraction quality.

Start the actual GPU run explicitly:

```sh
nix develop -c just train-full
```

The runner refuses an existing output directory unless resumed explicitly:

```sh
nix develop -c just resume-full
```

The configured output directory contains `status.json`, the frozen plan/config,
periodic checkpoints, `epoch-N-coverage.json`, and full Dev predictions/metrics.
`diagnostics.json` and `epoch-N-diagnostics.json` retain loss numerators and
denominators, proposal recall before gold injection, injected spans and missing
assignment targets, grouped by language, domain, source and output form. Those
counters resume with the optimizer and coverage state. Dev additionally records
span-stage diagnostics from the ordinary production invocation without injecting
gold. Exact source-boundary mismatches remain diagnostic and do not override
correct normalized values or reviewed answer alternatives.
Every completed epoch must prove that every accepted request was seen exactly once.
Mid-epoch checkpoints retain optimizer state, random state, cursor and coverage.
An interruption remains incomplete, and resume requires unchanged data and code.
`configs/active-training.json` records the actual run status too.

Only complete epochs may become `best.json`, using the normalized production
pipeline and the existing field/domain/rejection regression guard. Completing
training does not imply passing the general reliability gate or authorize release.
The checkpoint stores adapters and heads, with an exact binding to the merged
BiQwen base; it must not be loaded on the original Qwen base. The same API can load
these local checkpoints with `tractor serve --model <checkpoint-directory>`.

For a new source revision, rebuild with `just prepare-biqwen-full` into a fresh
output directory (the module also accepts `--output`), then bind its manifest hash
and exact counts in both configuration files. A prepared corpus or a passing CPU
software test is not a completed GPU training run.
