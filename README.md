# tractor · System One Extraction Model

![tractor — System One Extraction Model](docs/brand/tractor.svg)

**Text in. Structured data out. Ready for your application.**

tractor is Mayflower’s **System One Extraction Model**: a specialized model and API
for turning German and English text into product fields, separate records and
normalized quantities. Describe the fields you need. Get JSON your code can use.

Built for the extraction calls inside catalog imports, supplier integrations and
operational workflows, tractor combines learned language understanding with exact
unit conversion and an explicit API contract. Run it on your own infrastructure,
with CPU or GPU inference and an HTTP API that works with any language.

[User manual](docs/manual/index.md) · [Get started](#start-the-api) · [API](docs/api.md) ·
[Benchmarks](docs/benchmarks.md) · [Model card](release/MODEL_CARD.md)

## Put extraction on the fast path

| Your application needs | tractor provides |
|---|---|
| Fields from messy text | Source-span extraction with field descriptions in plain language |
| Consistent measurements | Locale-aware parsing and deterministic Decimal unit conversion |
| Multiple products in one request | Anchored records with values associated with each product |
| Explicit outcomes | Missing-value, normalization and ambiguity issues |
| Control over deployment | Local, offline serving with CPU or GPU support |
| A small integration surface | One HTTP endpoint and a lightweight Python SDK |

A single encoder pass finds the relevant spans. Code assembles the response and
normalizes quantities. That focused path makes tractor a practical alternative to
calling Haiku 5.5 or GPT-6 Luna for repeated, well-defined extraction tasks.
See the [evaluation results and supported scope](docs/benchmarks.md).

## Specialized extraction, measured

**94.41% complete-response accuracy. 330 ms median response time on CPU.**

On 680 identical synthetic extraction requests, tractor was **4.3× faster
than Haiku 5.5** and **5.7× faster than GPT-6 Luna** by median HTTP response
time, with higher complete-response accuracy than both prompt-only baselines.

![Extraction accuracy and median response times](docs/brand/comparison.svg)

CPU loopback versus remote cloud APIs, reasoning disabled, no retries or answer
repair. Results apply to this extraction contract and synthetic workload; they
include network overhead and do not measure optimized LLM pipelines.
[Full results, rounding-tolerant scores and reproducible evidence](docs/benchmarks.md).

## From supplier text to application data

A supplier writes **“Die Breite beträgt 12,5 cm.”** Your system needs millimeters.
tractor returns **`125`** under your field name. For multiple products, it returns
separate records. Structured issues identify missing or ambiguous values for review.

Use it to populate catalog attributes, normalize dimensions from short technical
specifications, or provide an extraction step inside a larger agent workflow.
The supported scope is short German/English text within a 512-token joint
text/schema budget; validate your own workload before automating decisions.

## Start the API

Use the Nix flake development environment. Authenticate with `hf auth login`
using an account with access to the private model repository, then download the
bundle and start the API:

```sh
nix develop
just setup
uv run --no-sync hf download mayflowergmbh/tractor --local-dir models/tractor
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
    )
    print(result.data)
```

The result contains separate Alpha and Beta records with widths 100 and 120 mm.
Details retain original character offsets. Missing quantities produce
`null` and a structured issue; competing candidates are all returned with `ambiguous_value`. The SDK imports no ML libraries; `just setup-sdk`
installs its development dependencies without the model stack.

## Evaluation and limits

On 2,048 synthetic acceptance requests, 92.68% of complete responses are correct
(Field F1 0.9876). A separate 20,000-request synthetic suite reaches 98.05%
(Field F1 0.9974). All 32 authored API regression cases pass. These datasets have
different distributions; see [methodology and limitations](docs/benchmarks.md).

Real-world accuracy is unmeasured. Long record lists, wide schemas and competing
values can still fail. The joint text/schema budget is 512 tokens. Unsupported or
ambiguous values are rejected rather than guessed. CPU response times and the cloud API comparison are
reported above; GPU latency has not been measured for this bundle.

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

Source: [mayflower/tractor](https://github.com/mayflower/tractor).
Model: [mayflowergmbh/tractor](https://huggingface.co/mayflowergmbh/tractor). Both repositories are
private during release preparation. Release licensing is not yet selected.
