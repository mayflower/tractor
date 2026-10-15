# Contributing to tractor

tractor is the System One Extraction Model and its local extraction API.
Use `nix develop` followed by `just setup` and `just check`. The CLI and Python
package remain `tractor`. Read [architecture](docs/architecture.md) before
changing extraction or normalization behavior.

Include a minimal request, expected result and actual result in bug reports.
Use synthetic examples when source documents contain private information.
Changes to quantities must preserve Decimal arithmetic, original source offsets,
explicit ambiguity handling and the SDK's independence from ML dependencies.

Use dev data for training decisions, checkpoint selection and prompt development.
Keep test data frozen. Report synthetic, authored and real-world measurements
separately. A changed benchmark prompt or configuration starts a new experiment;
retain failures and previous runs. Do not replace measured results with estimates.

Small fixes need focused checks; changes to native inference also need the real
HTTP/model smoke checks documented in the README. Include commands and outcomes
in pull requests. Do not commit model weights, API keys or raw local logs.
