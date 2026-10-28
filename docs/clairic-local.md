# Running the local Clairic-compatible profile

This serves a local Tractor/BiQwen bundle behind Cloudflare's Workers AI route
for Clairic. It is a documented **subset**: transport, schema and the official
SDK work; Clairic's model quality, probabilistic null decision, estimates,
forecasts and images are not reproduced. The contract is in
[clairic-compatibility.md](clairic-compatibility.md), the evidence in
[clairic-implementation-status.md](clairic-implementation-status.md).

## 1. Install and prepare the model (explicit, online)

```sh
nix develop
just setup                 # all extras, pinned in uv.lock
just prepare-biqwen        # pinned merged BiQwen base (downloads once, verified by hash)
```

Nothing below downloads models. The runtime never fetches weights or data.

## 2. Export and verify an inference bundle

A training checkpoint references its base by absolute path and contains
optimizer state. Export the explicit inference selection instead:

```sh
uv run --no-sync tractor export-bundle \
  --checkpoint /path/to/run/checkpoints/step-00004000 \
  --output models/tractor-biqwen-inference \
  --commit "$(git rev-parse HEAD)"
uv run --no-sync tractor verify-bundle --model models/tractor-biqwen-inference
```

The bundle contains `base/` (the pinned merged base, hashed against
`src/tractor/biqwen-lock.json`), `checkpoint/adapter/`,
`checkpoint/heads.safetensors` and `inference-bundle.json` (format, Tractor
commit, decoder, schema, attention contract, every file hash and the hash of the
original `checkpoint.json`). It has no `training-state.pt`, datasets, logs or
private paths, and it loads after being moved anywhere. An incomplete BiQwen
directory fails; it is never loaded as GLiNER.

## 3. Start the server (loopback)

Process:

```sh
uv run --no-sync tractor serve --api clairic \
  --model "$PWD/models/tractor-biqwen-inference" \
  --device cpu --precision fp32 --host 127.0.0.1 --port 8000
```

Optional local policy (not Clairic request keys): `--locale de-DE|en-US|en-GB`,
`--date-order DMY|MDY|YMD`, `--enable-compute`, `--uncertainty`.

Container (CPU, BiQwen extra, read-only model, one worker, loopback on the host):

```sh
docker build --build-arg TRACTOR_EXTRA=biqwen -t tractor-biqwen .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/models/tractor-biqwen-inference:/model:ro" tractor-biqwen \
  --api clairic --model /model --device cpu --host 0.0.0.0 --port 8000
```

`0.0.0.0` applies inside the container only. GPU or memory requirements are not
promised from the parameter count; measure them on the target host.

## 4. curl

```sh
curl --fail-with-body \
  http://127.0.0.1:8000/client/v4/accounts/local/ai/run/@cf/cloudflare/clairic \
  -H 'Content-Type: application/json' \
  --data-binary @examples/clairic/invoice.json
```

The request asks only for explicitly stated values. Expected shape with a
working model (values depend on the model; spans index the `state`):

```json
{"result": {"model": "clairic",
            "answers": {"invoice_number": {"type": "text", "value": "INV-2291", "span": [16, 24]},
                        "total": {"type": "number", "value": 1080, "span": [40, 48]}},
            "truncated": false,
            "usage": {"input_tokens": 120, "output_tokens": 0, "passes": 1},
            "state_text": "Invoice number: INV-2291. Total amount: 1080 USD."},
 "success": true, "errors": [], "messages": []}
```

## 5. Official Python SDK

```python
from langchain_cloudflare.extraction_models import CloudflareWorkersAIExtractionModel

model = CloudflareWorkersAIExtractionModel(
    account_id="local",
    api_token="local",
    api_base_url="http://127.0.0.1:8000/client/v4/accounts",
)
result = model.extract(
    state="Invoice number: INV-2291.",
    fields={"invoice_number": {"type": "text",
                               "instructions": "Extract the explicitly stated invoice number."}},
    options={"return_state_text": True},
)
print(result["answers"])
```

Tested with langchain-cloudflare 0.4.3 at revision
`85a0e443d618beac451335a3fcc423da211ae5c5` (`uv sync --extra clairic-conformance`).
The dummy `account_id`/`api_token` only satisfy the client; there is no
Cloudflare authentication. Passing this example proves transport integration,
not model quality or uncertainty semantics.

## 6. Limits, errors and capability requirements

- One pass, at most 32 internal fields (compute operands count), the model's
  token limit and a 1 MiB body. Larger requests fail; nothing is truncated.
- Instructions must use the documented forms; others return
  `tractor.unsupported_operation` before any inference.
- `estimate`, `forecast`, `images`, `reference_date` with date fields, and an
  explicit `null_threshold` without released uncertainty return 422 before
  inference.
- Invalid or ambiguous evidence returns `tractor.unresolved_field` with the
  native issue codes in `details`; it is never reported as `null`.
- `GET /v1/capabilities` shows each capability's evidence levels.

## 7. Without and with uncertainty artifacts

Without artifacts (every current bundle): no `p_null`, `confidence` or `guess`
keys; `null` means no accepted extraction value. With `--uncertainty`, the server
loads the bundle's uncertainty artifact only if it is bound to exactly this
bundle (weights, base, schema, decoder, projection/compiler versions,
precision), is not a fixture and was released against approved criteria. Then
scalar answers carry `p_null` (probability of *not* typed-answerable),
`confidence` (calibrated correctness of the returned answer, at the calibrated
threshold scope) and `guess`. No such artifact exists yet.

## 8. Open real checks and their commands

```sh
# Cold offline start in an isolated network namespace, real request, JSON report:
scripts/clairic-offline-check.sh models/tractor-biqwen-inference cpu fp32 reports/clairic/offline-check.json
# Software suite including the official client:
uv sync --locked --all-extras && uv run --no-sync pytest -q tests/test_clairic_*.py \
  tests/test_inference_bundle.py tests/test_uncertainty.py tests/test_numeric_distribution.py
# Uncertainty-head preflight (no 4B weights, no training):
uv run --no-sync python -m tractor.uncertainty_training --config configs/uncertainty-heads-dev.json
```

Real uncertainty training, calibration and estimate training need data that
does not exist yet and an explicit GPU authorization; see the status document.
