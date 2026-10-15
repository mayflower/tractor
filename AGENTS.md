# tractor

tractor is the System One Extraction Model and its local HTTP API/Python SDK.
The sole release bundle is models/tractor. Keep publication files focused on the
current product; do not add experiment histories, predecessor models or local paths.
Both release targets must remain private: mayflower/tractor and mayflowergmbh/tractor.

Use nix develop, just setup and uv run --no-sync. just check runs lint, fixtures
and unit tests. just check-model models/tractor cpu checks real HTTP and encoder
behavior. Keep SDK imports independent of ML libraries and downloads explicit.
Preserve Decimal arithmetic, original offsets, ambiguity handling and bounded
inference admission. Do not infer model quality from fixture-only tests.

Only dev data may select checkpoints or thresholds. Report synthetic and real-world
results separately. Do not run GPU latency measurements without a new user request;
the GPU is shared. Published latency measurements are CPU HTTP round trips only.

Create distributable snapshots with scripts/package-release.py. Its allowlists
exclude Git history, logs, datasets, training runs and private provenance. Do not
upload files or change repository visibility as part of packaging.
