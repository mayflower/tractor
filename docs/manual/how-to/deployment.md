# How to deploy tractor

Download the complete bundle first. The container expects it as a read-only mount;
it does not download weights or train during startup.

Address components require libpostal 1.1.4 and the `postal` Python bindings.
`nix develop` supplies the pinned C library and model data; `just setup` installs
the locked Python dependencies. The Docker build includes the same library revision
and data, verified against `configs/libpostal-data.sha256`. Installation downloads
several gigabytes of address-model data once. Runtime address parsing is local on
CPU, with no external service calls or GPU requirement.

## Run on CPU with Docker

```sh
docker build -t tractor:local .
docker run --rm --user "$(id -u):$(id -g)" -p 127.0.0.1:8000:8000 \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -v "$PWD/models/tractor:/model:ro" tractor:local
```

The image starts one worker on CPU and has a readiness health check. The UID/GID
option allows the container to read a bundle owned by the current user; make sure
its directories are traversable and files readable by that identity.

## Run with Compose

```sh
TRACTOR_UID="$(id -u)" TRACTOR_GID="$(id -g)" \
  docker compose -f examples/manual/compose.yaml up --build
```

The [Compose file](../../../examples/manual/compose.yaml) resolves its build context
and model mount relative to the example file, so it uses the repository root and
`models/tractor`. The UID/GID variables let it read owner-only model files without
changing their permissions. Stop it with the same command followed by `down` instead of
`up --build`. It binds the published port to localhost. The image's own health check
remains enabled.

## Use a GPU

With a working NVIDIA driver and NVIDIA Container Toolkit, add `--gpus all` and
explicit server arguments:

```sh
docker run --rm --gpus all --user "$(id -u):$(id -g)"   -p 127.0.0.1:8000:8000 -v "$PWD/models/tractor:/model:ro" tractor:local \
  --model /model --device cuda --precision fp32 --host 0.0.0.0 --port 8000
```

`bf16` requires CUDA hardware with BF16 support; `fp16` requires CUDA. Validate
accuracy when changing precision. Published release comparison timings are CPU
measurements and do not predict GPU latency.

## Run on Kubernetes

The Helm chart in `deploy/helm/tractor` runs one CPU worker per pod. An init
container downloads the pinned Hub revision into a persistent volume and verifies
the weights' SHA-256 before the server starts; the server itself stays offline.
Provide a Hugging Face read token in a Secret (key `HF_TOKEN`, name from
`model.tokenSecretName`) and, for a private image package, `imagePullSecrets`.
The Service is cluster-internal because the API has no authentication. Size memory
for the model plus the Python runtime; the defaults request 3 GiB and allow 6 GiB.

```sh
helm upgrade --install tractor deploy/helm/tractor --namespace tractor --create-namespace \
  --set image.tag=sha-COMMIT
```

## Operate the service

Use `/healthz` for process liveness and `/readyz` for inference readiness. Route
traffic only to ready instances. An internal inference error marks a server unready;
a supervisor must restart it after investigation. The CLI does not offer hot reload
or automatic model replacement. Shut down gracefully so admitted work can finish.

The service has no built-in TLS, authentication, rate limiting by caller, or CORS.
Keep it on a trusted network or behind your authenticated gateway. Configure the
proxy's request timeout for your workload and preserve non-2xx status codes. The
API's 128 KiB request-body limit and joint 512-token model window are separate limits.

Inspect `Server-Timing` for queue, preprocessing, inference, and normalization. Its
inference stage includes native preprocessing/decoding; it is not a GPU-kernel-only
measurement. Standard server logs avoid source text, but your proxy and application
logging policies must be configured separately.

[Manual home](../index.md)
