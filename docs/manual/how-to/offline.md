# How to install an offline or pinned model

On a machine with network access, download a specific Hub revision using an account
with repository access. Find the verified model revision in
[release targets](../../../release/targets.json).

```sh
revision=$(jq -r '.huggingface_revision' release/targets.json)
uv run --no-sync hf download mayflowergmbh/tractor \
  --revision "$revision" --local-dir models/tractor
sha256sum models/tractor/model.safetensors
```

The released weights must have SHA-256:

```text
f9b177aa18d1f968cf8e2b3a99cff8938c096be8570064182f84c5a16488b90a
```

Transfer the whole model directory to the serving machine. Copy the tokenizer,
encoder configuration, thresholds, and normalization policy together with the
weights. A model card or a Git LFS pointer file is not the weights. Use `hf download`
instead of copying a pointer from a source-only Git checkout.

Install locked runtime dependencies or transfer a prebuilt container image before
disconnecting the serving host. Nix and uv may require downloads during environment
creation; offline model loading alone does not make an unprepared development
machine fully offline.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync tractor serve \
  --model models/tractor --device cpu
```

The loader also enables offline mode internally. No inference request needs an
external model API. Do not run `tractor prepare-model` to install the released
fine-tuned bundle: that command prepares the upstream base model for development.

For an upgrade, download to a new directory, validate your held-out integration
cases, start a separate server, and switch traffic after readiness. Preserve the
previous deployment bundle until the new one is accepted. Keep weights and
normalization code compatible; the loader rejects a mismatched policy.

[Manual home](../index.md)
