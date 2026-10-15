# tractor · System One Extraction Model

![tractor — System One Extraction Model](docs/brand/tractor.svg)

Extract structured fields and normalized quantities from German and English text.
tractor runs locally, selects original source spans with GLiNER, and converts
quantities with deterministic Decimal arithmetic. It includes an HTTP API and a
lightweight Python SDK.

[API](docs/api.md) · [Model card](release/MODEL_CARD.md) ·
[Evaluation](docs/benchmarks.md) · [Contributing](CONTRIBUTING.md)

## Start the API

Use the Nix flake development environment and place the model bundle in
`models/tractor`:

```sh
nix develop
just setup
tractor serve --model models/tractor --device cuda
```

Use `--device cpu` for CPU inference. Serving loads local files offline and does
not download or train a model. `GET /readyz` reports readiness; interactive API
documentation is available at `http://127.0.0.1:8000/docs`.

```sh
curl http://127.0.0.1:8000/v1/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"Die Breite beträgt 12,5 cm.","locale":"de-DE","fields":{"width_mm":{"type":"quantity","description":"Breite","unit":"mm"}}}'
```

```json
{"data":{"width_mm":125},"issues":[]}
```

## Python SDK

```python
from tractor import Client, Quantity, Text

with Client("http://127.0.0.1:8000") as client:
    result = client.extract(
        "Alpha: Breite 10 cm. Beta: Breite 12 cm.",
        {"product": Text("Produktbezeichnung"),
         "width_mm": Quantity("Breite", unit="mm")},
        anchor="product",
        many=True,
        include_details=True,
    )
    print(result.data)
```

The result contains separate Alpha and Beta records with widths 100 and 120 mm.
Details retain original character offsets. Missing or ambiguous quantities produce
`null` and structured issues. The SDK imports no ML libraries; `just setup-sdk`
installs its development dependencies without the model stack.

## Evaluation and limits

On 2,048 synthetic acceptance requests, 92.68% of complete responses are correct
(Field F1 0.9876). A separate 20,000-request synthetic suite reaches 98.05%
(Field F1 0.9974). All 32 authored API regression cases pass. These datasets have
different distributions; see [methodology and limitations](docs/benchmarks.md).

Real-world accuracy is unmeasured. Long record lists, wide schemas and competing
values can still fail. The joint text/schema budget is 512 tokens. Unsupported or
ambiguous values are rejected rather than guessed. No validated latency figure is
published for this bundle.

## Development

```sh
nix develop
just setup
just check
just check-model models/tractor cpu
```

Python dependencies and native libraries are pinned in `uv.lock` and `flake.lock`.
The CLI also provides data generation, validation, training, evaluation and
benchmark commands; run `tractor --help`. See [architecture](docs/architecture.md)
and [upstream compatibility](docs/upstream-compatibility.md).

## Container

```sh
docker build -t tractor:local .
docker run --rm --user "$(id -u):$(id -g)" -p 127.0.0.1:8000:8000 \
  -v "$PWD/models/tractor:/model:ro" tractor:local
```

The container defaults to CPU. GPU serving requires NVIDIA Container Toolkit;
add `--gpus all` and append `--model /model --device cuda --host 0.0.0.0`.

Source: `mayflower/tractor`. Model: `mayflowergmbh/tractor`. Both repositories are
private during release preparation. Release licensing is not yet selected.
