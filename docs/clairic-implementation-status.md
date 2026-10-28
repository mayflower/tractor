# Local Clairic profile: implementation status and acceptance

- Integration branch: `experiment/biqwen-extract-decide-4b`
- Starting commit: `a10087992dce9d0793eb392f8e1e6ebbacb5dccf` (identical to the
  researched reference)
- Date: 2026-10-28
- Specification: [clairic-compatibility.md](clairic-compatibility.md); operation:
  [clairic-local.md](clairic-local.md)

This is the versioned record, not a live process status. No training, GPU
work, external inference, push, publication or deployment was performed.
`configs/active-training.json`, the running full-training configuration, base
weights, datasets and checkpoints were not modified.

## Architecture decision

The profile extends Tractor/BiQwen. The request compiler maps Clairic fields
onto existing typed Tractor contracts; extraction, normalization and ambiguity
handling are the production core; a projection writes the Workers AI payload.
Uncertainty heads, outcome scorer, calibration and the numeric distribution
head are separate, versioned architecture parts that are disabled until real
artifacts are trained, calibrated, bound and released. Neither an identical
Clairic implementation nor equal model quality is claimed or shown.

## Changed files

| Area | Files |
|---|---|
| Protocol, capabilities, errors (01) | `src/tractor/clairic_types.py`, `src/tractor/capabilities.py` |
| Evidence (02) | `src/tractor/evidence.py`; observing hooks in `document_pipeline.py`, `service.py` (`extract_traced`), raw signals in `biqwen.py` (`candidates_with_signals`) |
| Compiler (03) | `src/tractor/clairic_compiler.py` |
| REST profile, projection (04) | `src/tractor/server.py` (shared `InferenceWorker`, `api=`), `src/tractor/clairic_server.py`, `src/tractor/clairic_results.py`, `src/tractor/cli.py` |
| Conformance (05) | `tests/test_clairic_conformance.py`, `tests/fixtures/clairic_official/` (vendored MIT helpers), `pyproject.toml`/`uv.lock` extra `clairic-conformance` |
| Bundle, offline (06) | `src/tractor/inference_bundle.py`, `src/tractor/backends.py`, `biqwen_training.py` (`apply_checkpoint` split out of `load_checkpoint`), `Dockerfile` (`TRACTOR_EXTRA`), `scripts/clairic-offline-check.sh` |
| Uncertainty (07–09) | `src/tractor/uncertainty.py`, `uncertainty_heads.py`, `uncertainty_training.py`, `uncertainty_calibration.py`, `uncertainty_artifacts.py`, `configs/uncertainty-heads-dev.json` |
| Compute (10) | `src/tractor/clairic_compute.py` |
| Estimate (11) | `src/tractor/numeric_grid.py`, `src/tractor/numeric_distribution.py` |
| Contract, docs (00, 12) | `AGENTS.md`, `reqs.md` (narrow amendments), `docs/clairic-*.md`, `examples/clairic/invoice.json` |
| Tests | `tests/test_clairic_{contract,evidence,compiler,http,compute,conformance}.py`, `tests/test_inference_bundle.py`, `tests/test_uncertainty.py`, `tests/test_numeric_distribution.py`, `tests/clairic_backend.py` |

## Implemented behavior

- `tractor serve --api tractor|clairic`; `tractor` is the default and one profile
  runs per process. Both use one model and one serialized worker (lock held until
  the worker finishes, bounded queue, wait timeout, cancellation shielding,
  failures mark the model unready without logging source text).
- Strict parsing of all nine Clairic types, options and transport hints;
  missing vs explicit `null_threshold` distinguished.
- Capability gates with separate evidence levels (architecture, weights,
  provenance, calibration, release) and bindings; current bundles declare only
  `source_spans`, `typed_extraction`, `typed_lists`.
- Compile-time rejection (no inference) of estimate, forecast, images,
  `reference_date` with dates, explicit thresholds without `p_null`, and
  instructions outside the documented forms; mixed requests fail as a whole.
- Request-scoped evidence: occurrences, raw BiQwen source logits/temperature/
  scores and assignment margins from the same pass, normalized values with all
  supporting occurrences, rejected occurrences with codes.
- Projection: exact text spans, typed numbers/dates/durations under explicit
  precision rules, list values plus occurrence spans, errors for invalid or
  ambiguous evidence, honest usage, `truncated: false` only because nothing is
  truncated.
- Threshold rule `value is null ⇔ p_null > threshold` with explicit
  inconsistency error, active only with a released provider.
- Deterministic compute (sum, difference, product, date + constant) behind
  `--enable-compute`, one extraction pass for all operands.
- Relocatable inference bundle: explicit file selection, full hashes, no
  training state, verification without ML imports, loader without access to the
  training folder; incomplete BiQwen layouts never load as GLiNER.
- Uncertainty software: heads on field queries (not registered in the
  extraction state dict, never emitting without release), masked labels from
  gold states, preflight, Platt calibration with group-separated
  fit/select/validate partitions, p_null-first then outcome calibration,
  reliability/ECE/risk-coverage metrics, artifacts bound to the exact bundle.
- Estimate prototype: exact Decimal/log/date grids, one-softmax head with tail
  states, CDF quantiles, two-hot loss; disabled.

## Checks executed

Run on CPU in a scratch environment (Python 3.12; pydantic, FastAPI, uvicorn,
httpx, Babel, Pint, dateparser, CPU torch, safetensors, the pinned official
client). libpostal and transformers/peft/gliner2 were **not** installed;
libpostal was stubbed locally for import only.

| Check | Result |
|---|---|
| New tests (`test_clairic_*`, `test_inference_bundle`, `test_uncertainty`, `test_numeric_distribution`) | 185 passed, 1 skipped (level B, unsupported by design) |
| Official client level A: sync and async over a real loopback uvicorn server with external connections blocked | passed |
| Official helper level B on the subset | unsupported (fails on missing `confidence`/`p_null`, as expected) |
| Level B projection invariants with fixture probabilities | passed; fixture only, not calibration |
| Full existing suite, before vs after, same environment | before 751 passed / 76 failed; after 939 passed / 73 failed; **no test newly failing**. Remaining failures and 3 collection errors need transformers, gliner2 or libpostal data that this environment lacks |
| `ruff check src scripts tests examples` | clean |
| `scripts/check-docs.py`, `tests/fixtures/check_fixtures.py`, `uv lock --check` | passed |

`just check` and `just check-model` were not run in the full Nix environment.

## Acceptance matrix

| Capability | implemented | software test | real inference | quality | calibration | released | concrete open evidence |
|---|---|---|---|---|---|---|---|
| Route, envelope, alias, profiles | yes | passed | not executed | — | — | yes (subset) | offline check with a real bundle |
| Official SDK transport (level A) | yes | passed | not executed | — | — | yes | same, with a real bundle |
| Text / number / date / time / datetime / duration extraction | yes | passed | not executed | not executed | no | yes (subset) | Clairic-style Dev evaluation on the production path |
| Typed lists | yes | passed | not executed | not executed | no | yes (subset) | same |
| Evidence spans and raw signals | yes | passed | not executed | — | — | internal | real BiQwen pass through `candidates_with_signals` |
| Relocatable bundle, offline start | yes | passed | not executed | — | — | — | `scripts/clairic-offline-check.sh` with an exported checkpoint |
| Deterministic compute | yes | passed | not executed | not executed | no | opt-in flag | evaluation of the documented forms |
| `p_null` (typed answerability) | software only | passed (fixtures) | not executed | not executed | not executed | no | labeled data, head training, calibration, approved criteria |
| `confidence` (final outcome) | software only | passed (fixtures) | not executed | not executed | not executed | no | out-of-fold predictions, calibration |
| `source_presence` | software only | passed (fixtures) | not executed | not executed | not executed | no | same |
| Official uncertainty invariants (level B) | projection only | unsupported | not executed | — | — | no | released artifacts |
| `estimate` | prototype | passed | not executed | not executed | not executed | no | estimate dataset, training, interval evaluation |
| `forecast` | no | parse only | — | — | — | unsupported | own design and evidence |
| `images` | no | parse only | — | — | — | unsupported | local vision path; external OCR would not be local |

## Missing data, weights and real evidence

1. Real CPU inference and the cold offline proof with an exported bundle of a
   finished checkpoint (`scripts/clairic-offline-check.sh`). The active full
   training has not finished, and no bundle exists on this machine.
2. Evaluation of Clairic-style requests on the full production path; current
   quality evidence is for the native contract only.
3. Uncertainty: `data/uncertainty-heads-v1/*` does not exist yet; out-of-fold
   final-outcome predictions are needed; head training needs an explicit GPU
   authorization and must not run next to the active training; calibration
   needs the separate calibration split; release criteria are proposals
   (e.g. validate ECE ≤ 0.05 per event, ≥ 200 fields and ≥ 20 per class) and
   are **not** decided.
4. Estimate: a separate dataset contract with numeric targets, bounds and
   provenance; training; evaluation of error by magnitude, interval coverage
   and width.

## Deviations from the researched plan

- Field batching across passes is not implemented: requests above the local
  limits fail instead (documented in the spec).
- `reference_date` is rejected with temporal fields instead of completing years.
- The uncertainty threshold scope is checked inside the provider (after
  inference) rather than at compile time; it still never returns probabilities
  outside the calibrated scope.
- The estimate head reads a supplied field representation; wiring it into the
  BiQwen decoder is left until a trained head exists.
- The level-B check vendors the official assertion helpers (MIT) because they
  are not part of the installed package.
