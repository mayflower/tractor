# Tutorial: your first extraction

You will start a local server and turn a German width measurement into millimeters.
Use a Linux machine with Nix flakes enabled and access to the private GitHub and
Hugging Face repositories. CPU inference is enough for this tutorial. Initial
installation downloads Python dependencies and the model; allow several GB of disk
space. The flake supplies Python 3.12, uv, curl, jq, and just.

## 1. Install the environment

```sh
git clone https://github.com/mayflower/tractor.git
cd tractor
nix develop
just setup
```

`just setup` installs the full serving and training environment from the lock file.
Keep using `uv run --no-sync` for commands so a client invocation does not remove
installed server extras.

## 2. Download the released bundle

```sh
uv run --no-sync hf auth login
uv run --no-sync hf download mayflowergmbh/tractor --local-dir models/tractor
```

Log in interactively with an account that can read the repository. The complete
bundle includes weights, tokenizer, encoder configuration, thresholds, and the
normalization policy. Serving a weights file alone is insufficient.

## 3. Start the API

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync tractor serve \
  --model models/tractor --device cpu
```

Leave this terminal running. In a second terminal, enter the same checkout and run
`nix develop`. Wait until the server reports that startup is complete, then check:

```sh
curl --fail-with-body http://127.0.0.1:8000/readyz
```

You should see `{"status":"ready"}`. Startup loads the local bundle and performs
internal warmups; readiness is enabled afterward. Loading is slower than a normal
extraction request.

## 4. Extract a quantity

The file [scalar.json](../../../examples/manual/scalar.json) asks for the width in mm:

```json
{"text":"Breite: 12,5 cm.","locale":"de-DE","fields":{"width_mm":{"type":"quantity","description":"Breite","unit":"mm"}}}
```

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/extract \
  -H 'Content-Type: application/json' \
  --data-binary @examples/manual/scalar.json
```

Expected result:

```json
{"data":{"width_mm":125},"issues":[]}
```

The field name `width_mm` belongs to your application. `description` tells the
model which property to find; `unit` tells the normalizer which output unit to use.

## 5. Inspect an incomplete result

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"Modell CM125 hat keine Maßangaben.","fields":{"width_mm":{"type":"quantity","description":"Breite","unit":"mm"}}}'
```

Expected result:

```json
{"data":{"width_mm":null},"issues":[{"field":"width_mm","code":"not_found"}]}
```

This is a successful HTTP request with an unresolved field. Your application can
send it for review. A missing value does not mean zero.

Open `http://127.0.0.1:8000/docs` to inspect the API. Stop the server with Ctrl-C
when finished. Continue with [the Python tutorial](python.md).

[Manual home](../index.md)
