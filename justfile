default:
    @just --list

# Experimental BiQwen backend, same public API. Explicit download, pinned revision.
prepare-biqwen:
    uv sync --locked --extra biqwen --extra dev
    uv run --no-sync python scripts/prepare-biqwen.py

serve-biqwen:
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync tractor serve --model models/biqwen-extract-decide-4b --device cuda --precision bf16 --port 8002

# Full corpus only. Preparation/preflight never start GPU training.
prepare-biqwen-full:
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false uv run --no-sync python -m tractor.biqwen_training_data

check-biqwen-full:
    uv run --no-sync python -m tractor.biqwen_training --config configs/biqwen-full.json

train-full:
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false uv run --no-sync python -m tractor.biqwen_training --config configs/biqwen-full.json --train

resume-full:
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false uv run --no-sync python -m tractor.biqwen_training --config configs/biqwen-full.json --train --resume

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
    uv run --no-sync python scripts/check-docs.py
    uv run --no-sync pytest -q tests

doctor:
    uv run --no-sync python scripts/check-env.py

doctor-training:
    uv run --no-sync tractor doctor --load

check-native-tasks model="models/tractor":
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync python scripts/check-episodes.py --model {{model}}

# Real HTTP and encoder checks against a local model.
check-model model="models/tractor" device="cuda":
    uv run --no-sync python scripts/http-smoke.py --model {{model}} --device {{device}}
