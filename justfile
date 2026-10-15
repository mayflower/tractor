default:
    @just --list

# Lightweight SDK dependencies and developer tools.
setup-sdk:
    uv sync --locked --extra dev

# Full local inference/training environment (large CUDA wheel downloads).
setup:
    uv sync --locked --all-extras

check:
    uv lock --check
    uv run --no-sync ruff check src scripts tests examples
    uv run --no-sync python tests/fixtures/check_fixtures.py
    uv run --no-sync pytest -q tests

doctor:
    uv run --no-sync python scripts/check-env.py

# Real HTTP and encoder checks against a local model.
check-model model="models/tractor" device="cuda":
    uv run --no-sync python scripts/http-smoke.py --model {{model}} --device {{device}}
