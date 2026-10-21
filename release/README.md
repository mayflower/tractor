# Release packaging

Source repository: `mayflower/tractor` (private).
Model repository: `mayflowergmbh/tractor` (private).
Model bundle: `models/tractor`.

```sh
nix develop
just check
just check-model models/tractor cpu
uv run --no-sync python scripts/package-release.py
```

Packaging creates standalone source and model archives in `dist/`, with SHA-256
manifests. Source packaging uses an explicit allowlist and excludes Git history,
local reports, training datasets, checkpoints and secrets. The narrow synthetic comparison fixture and current pipeline Dev metrics are
included; no LLM comparison is claimed for the current weights. The model archive contains only
the current weights, tokenizer, configuration, normalization policy, source lock,
model card and evaluation metrics. Packaging does not upload anything.

Keep both remote repositories private. Update the existing source publication checkout with the clean snapshot; do not
push the development repository's existing history.
Use the model archive contents for Hugging Face. Source and model licensing remain
to be selected before public distribution; preserve upstream attribution.
