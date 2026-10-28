# Local Clairic-compatible profile: specification

Status: implemented as a documented local subset on branch
`experiment/biqwen-extract-decide-4b` (starting commit
`a10087992dce9d0793eb392f8e1e6ebbacb5dccf`). The current state of evidence is in
[clairic-implementation-status.md](clairic-implementation-status.md); operating
instructions are in [clairic-local.md](clairic-local.md).

Statements below are marked as one of:

- **[CF]** Cloudflare's published contract, from the pinned sources at the end.
- **[local]** a versioned local decision (`clairic-local-decisions-v1`).
- **[open]** an open question or missing evidence.

## 1. Goal

One server process serves exactly one API profile, chosen at startup:

```sh
tractor serve --api tractor --model /abs/path/to/bundle   # default, unchanged
tractor serve --api clairic --model /abs/path/to/bundle
```

Both profiles use the same loaded model, the same serialized inference worker
(one active job, bounded queue, wait timeout, lock held until the worker
really finishes) and the same production core: typed Tractor field contracts,
`service.resolve`, `document_pipeline.resolve_document` and the existing
normalizers. The Clairic profile adds a request compiler and a result projection
around that core; it is not a second extraction or normalization pipeline.

## 2. Unchanged native contract

`POST /v1/extract`, `Client.extract`, `ExtractRequest` validation and the public
`data`/`issues` response are unchanged. The native profile still exposes no
evidence, score or offset options. Instrumentation added for the Clairic
profile (a request-scoped evidence collector) only observes; tests assert that
`extract()` and `extract_traced()` return identical native results.

## 3. Route, alias and client configuration

- **[CF]** `POST /client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clairic`, request
  `model` is `clairic` or `@cf/cloudflare/clairic`; the REST response wraps the
  model payload as `{"result", "success", "errors", "messages"}`.
- **[local]** `account_id` is a local identifier matching `[A-Za-z0-9_-]{1,64}`; it is
  not a Cloudflare account. The alias always means the one bundle loaded at
  startup. No request can name a path or load weights.
- **[local]** There is no Cloudflare authentication. An `Authorization` header is
  accepted and ignored. The default bind address stays `127.0.0.1`; expose the
  port only behind an authenticating proxy.
- **[local]** `GET /v1/capabilities` (Clairic profile only) lists every capability with
  its evidence levels. `/healthz` and `/readyz` exist in both profiles.
- Official client: `CloudflareWorkersAIExtractionModel(account_id="local",
  api_token="local", api_base_url="http://127.0.0.1:8000/client/v4/accounts")`.
  Tested revision: langchain-cloudflare `0.4.3` at
  `85a0e443d618beac451335a3fcc423da211ae5c5` (extra `clairic-conformance`).

## 4. Types and operations

Legend: parse = request validated; run = executable in this build; trained /
calibrated = real model evidence exists; tested = software tests with
controlled backends (not model quality).

| Type / operation | parse | run | trained | calibrated | tested | notes |
|---|---|---|---|---|---|---|
| `text` (explicit value) | yes | yes | yes¹ | no | yes | value equals `state[span]` |
| `number` (explicit value) | yes | yes | yes¹ | no | yes | see unit rules (§6.4) |
| `date`, `time`, `datetime` | yes | yes | yes¹ | no | yes | unambiguous forms or server date order |
| `duration` | yes | yes | yes¹ | no | yes | exact unit conversion only (§6.5) |
| `list` of `text`/`number`/`date` | yes | yes | yes¹ | no | yes | typed multiple extraction |
| `p_null`, explicit `null_threshold` | yes | only with released artifact | no | no | yes (fixture) | rejected otherwise |
| `confidence` | — | only with released artifact | no | no | yes (fixture) | omitted otherwise |
| derived values (sum, difference, product, date + constant) | yes | with `--enable-compute` | n/a | no | yes | deterministic, §6.8 |
| other derivations in `instructions` | yes | no | no | no | yes | rejected as unknown |
| `estimate` | yes | no | no | no | prototype only | grid codec and head exist, disabled |
| `forecast` | yes | no | no | no | parse only | unsupported |
| `images` | yes | no | no | no | parse only | unsupported |
| `reference_date` with temporal fields | yes | no | no | no | yes | rejected (§6.6) |

¹ Through the existing BiQwen/GLiNER extraction training of the native product;
quality is not established for Clairic-style requests (see status document).

## 5. Compatible subset versus behavioral parity

**[CF]** The raw schema requires only `type` and `value` per answer; `p_null` and
`confidence` are optional. The official test helper, however, asserts both on
every answer.

**[local]** Without released uncertainty artifacts this server returns the subset:
probabilities are omitted (never `null`, never placeholder numbers), and every
request that needs them (an explicit `null_threshold`, even `0.5`) is rejected
before inference. A `value: null` in the subset means only that no extraction
value was accepted without a specific processing error. It is neither
Clairic's probabilistic null decision nor proof that the source lacks the value.
The default probabilistic null decision of Clairic is therefore **not**
reproduced in the subset.

Conformance levels are reported separately: **A** route, schema and SDK
transport; **B** the official uncertainty invariants. The subset meets A and
reports B as `unsupported`. With released artifacts, the projection applies
`value is null ⇔ p_null > null_threshold` (equality does not exceed; default
0.5) and fails with `tractor.projection_inconsistency` if `p_null` is at or
below the threshold but no valid value exists.

## 6. Local decisions (`clairic-local-decisions-v1`)

1. **Strict keys.** Unknown top-level keys, unknown option keys and keys not
   documented for a field's type are rejected. JSON `null` for an optional key is
   rejected, not treated as a default. `model` is required although the schema
   declares a default. `options.rejectIfBusy` (added by the official transport)
   is accepted and fails fast with `tractor.overloaded` when the worker is busy.
2. **Field names** follow the published pattern `^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$`
   and are returned unchanged; internal IDs are `f0, f1, …`.
3. **Field semantics.** The model receives a short description: the label of an
   accepted instruction form, or the field name with `_ . -` turned into spaces.
4. **Number units.** A currency code (`EUR GBP USD CAD AUD CHF`) selects the money
   normalizer; the source must state that currency, nothing is converted. A unit
   from Tractor's Pint list selects the quantity normalizer; the source must state
   a convertible unit. Any other unit is an output label; no conversion is claimed.
   Stated values are never rounded: more decimals than `decimals`
   (or any for `integer`) is `precision_exceeded`. Integral values are returned as
   JSON integers.
5. **Duration.** Values are converted from the ISO duration of the existing
   normalizer into the requested unit (default seconds) only when exact. Months
   and years have no fixed length and convert only between each other.
6. **Dates.** No implicit clock or time zone. `reference_date` would require
   year completion, which is not implemented, so it is rejected whenever a date,
   datetime or date list field is present.
7. **Locale.** Clairic has no locale key. A server option (`--locale`,
   `--date-order`) is the only source; without it only unambiguous forms parse.
8. **Instructions.** Accepted source-only forms (whitespace normalized):
   - absent instructions;
   - `Extract|Copy|Return the explicitly|literally stated|written|printed|given|mentioned <label>[ exactly as written][.][ Return null if (it is) absent|missing|not stated.]`;
   - `Extrahiere|Übernimm den|die|das ausdrücklich|wörtlich genannte(n)|angegebene(n)|geschriebene(n) <label>[.][ Gib null zurück, wenn er|sie|es fehlt.]`.

   `<label>` is one to eight words without digits and without derivation words
   (plus, including, expected, …). Accepted compute forms (only with
   `--enable-compute`):
   - `Sum|Difference|Product of the explicitly stated <a> and the explicitly stated <b>`;
   - `The explicitly stated <date> plus <n> days|weeks|months|years`.

   Everything else is `unknown` and rejected. This allowlist is deliberately small;
   e.g. `Amount due including shipping` or `Purchase order number` are rejected
   rather than answered with some number from the text.
9. **Compute.** Decimal arithmetic; result rounded once, half-even, to `decimals`;
   months/years add on the calendar with month-end clamping (Jan 31 + 1 month =
   Feb 28/29); missing, invalid or competing operands are errors; computed values
   have no span. No exchange rates, no implicit clock, no cross-record mixing (the
   profile has no records).
10. **Text and lists.** Text values are exact source spans. List values are unique
    normalized values in source order; `spans` lists every accepted occurrence,
    so lengths may differ and several spans may support one value. Numbers and
    dates carry the `span` of the first occurrence that produced the value: it is
    evidence, not string equality.
11. **Offsets** are half-open Python string indices (Unicode code points) into the
    unmodified `state`; CR/LF and whitespace are never normalized. This matches
    the official Python helper (`state[start:end]`). **[open]** Parity with UTF-16
    clients is not claimed.
12. **Errors.** Present but invalid or ambiguous evidence (`ambiguous_value`,
    `currency_mismatch`, `non_scalar_quantity`, …) is an error, never `null`.
    Error envelope: `{"result": null, "success": false, "errors": [{"code", "message",
    "field"?, "capability"?, "details"?}], "messages": []}` with local string codes:
    `tractor.invalid_request` 400, `tractor.unsupported_operation` 422,
    `tractor.capability_unavailable` 422, `tractor.unresolved_field` 422,
    `tractor.capacity_exceeded` 413/422, `tractor.overloaded` 503,
    `tractor.not_ready` 503, `tractor.projection_inconsistency` 500,
    `tractor.internal_error` 500. These are not Cloudflare error numbers.
13. **Capacity.** **[CF]** metadata says 65,536 context; the schema says fields
    are split across 8,192-token passes. **[local]** No splitting: one pass, at most
    32 internal fields (including compute operands), the backend's own token
    limit, 1 MiB body. Exceeding any limit is an error; input is never truncated
    (`truncated` is always `false`).
14. **Usage.** `input_tokens` is the length of the real encoded input including
    schema and special tokens; `output_tokens` is 0 (nothing is generated);
    `passes` is 1. `state_tokens` and `schema_tokens` are omitted because the
    tokenizer does not split them exactly. Not a billing count.
15. **Images.** **[CF]** the raw schema types `span` as an integer array while the
    official helper expects `span: null` with `source: "image"`. **[local]** Images
    are unsupported, so no decision is needed yet.
16. **Response `model`** is always `clairic`.

## 7. Planned and executed acceptance checks

| Check | Where | Status |
|---|---|---|
| Field names, type rules, defaults vs explicit null, NaN/duplicate keys | `tests/test_clairic_contract.py` | passing |
| Evidence preservation, offsets, signals, one pass | `tests/test_clairic_evidence.py` | passing |
| Compiler mapping, instruction forms, compile-time rejection | `tests/test_clairic_compiler.py` | passing |
| Route, envelope, errors, threshold rule, lifecycle | `tests/test_clairic_http.py` | passing |
| Compute arithmetic and calendar rules | `tests/test_clairic_compute.py` | passing |
| Official client level A over loopback | `tests/test_clairic_conformance.py` | passing |
| Official helper level B | same | unsupported |
| Relocatable bundle, broken bundles, no GLiNER fallback | `tests/test_inference_bundle.py` | passing |
| Uncertainty heads, labels, preflight, calibration, binding | `tests/test_uncertainty.py` | passing |
| Estimate grid/head prototype | `tests/test_numeric_distribution.py` | passing |
| Real BiQwen inference, offline cold start | `scripts/clairic-offline-check.sh` | not executed |
| Model quality on Clairic-style requests | — | not executed |

## 8. Sources (fixed revisions)

- Raw schema: cloudflare/cloudflare-docs@`e2dab6cf6f9038b866b96174cb4c447b2f834c10`,
  `src/content/workers-ai-models/clairic.json`
- Code samples and the real example response:
  same revision, `src/components/models/code/ClairicCode.astro`
- Client: cloudflare/langchain-cloudflare@`85a0e443d618beac451335a3fcc423da211ae5c5`,
  `libs/langchain-cloudflare/langchain_cloudflare/{extraction_models.py,_native_models.py}`
- Test helpers: same revision, `libs/langchain-cloudflare/tests/extraction_helpers.py`,
  `tests/unit_tests/test_extraction_models.py`,
  `examples/workers/src/system_one.py` (MIT; the assertion helpers are vendored
  unchanged in `tests/fixtures/clairic_official/` with the license)
- The Clef joint-schema code on Hugging Face is a research reference only; no code
  was taken from it, and nothing here claims Clairic's internal architecture.
