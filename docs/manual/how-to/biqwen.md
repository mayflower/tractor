# Use BiQwen behind the Tractor API

This experimental branch supports
[johannhartmann/biqwen-extract-decide-4b](https://huggingface.co/johannhartmann/biqwen-extract-decide-4b)
at revision `96184881677b03adbf28ed5ba18a17548f8fb861`. The local model directory
selects the backend. Requests, the Python SDK, field types, normalization and the
`data`/`issues` response remain the same.

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
compatibility/quality comparison, not final calibration or an independent test:
BiQwen was trained from a derivative of the Tractor corpus, and overlap with its
training split has not been ruled out. No latency measurement is part of this run.

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
