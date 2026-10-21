# HTTP API reference

Base URL defaults to `http://127.0.0.1:8000`. All extraction responses use JSON.

| Method and path | Purpose |
|---|---|
| `POST /v1/extract` | Extract one source text with a supplied field schema. |
| `GET /healthz` | Process liveness: `{"status":"alive"}`. |
| `GET /readyz` | `{"status":"ready"}`, or 503 `not_ready`. |
| `GET /docs` | Interactive API documentation. |
| `GET /openapi.json` | Generated request and response schema. |

## Request body

Send `Content-Type: application/json`. Unknown properties, duplicate JSON keys,
NaN/Infinity literals, and incompatible types are rejected.

| Property | Type/default | Constraints |
|---|---|---|
| `text` | required string | Nonempty and not all whitespace; original source text. |
| `fields` | required object | 1–32 named field definitions. |
| `locale` | string or null; null | `de-DE`, `en-US`, `en-GB`, or null. |
| `anchor` | string or null; null | Must name an existing text, name, place or identifier field. |
| `many` | boolean; false | True requires an anchor; controls array output. |

Field keys match `^[A-Za-z][A-Za-z0-9_]{0,63}$`. Each definition has a required
`type` (`text`, `quantity`, `number`, `money`, `date`, `address`, `identifier`, `list`, `person`, `organization`, `place`, `phone`, `email`, `postcode`, `time`, `datetime`, `duration`, `interval`, `percent`, `iban`, or `url`) and a 1–240 character `description`. A quantity also
requires `unit`, from the [canonical units](quantities-and-issues.md). Text fields
must not have a unit. Each type selects its processing; see the
[typed field parameters](../how-to/document-processing.md). There is no separate
processing map. Field definitions reject unknown properties. Booleans must
be JSON booleans, not strings such as `"true"`.

```json
{"text":"Alpha ist 10 cm breit. Beta ist 2 inches breit.","fields":{"product":{"type":"text","description":"Produktbezeichnung"},"width_mm":{"type":"quantity","description":"Breite","unit":"mm"}},"anchor":"product","many":true}
```

## Successful response

`data` is a field-keyed object, or an array of such objects when `many=true`.
Every record contains all requested keys. Values are source strings, normalized
JSON numbers, or null; list cardinalities return arrays of values. A scalar field with distinct competing candidates also returns an array, together with an `ambiguous_value` issue. Equivalent numbers may use exponent notation or trailing
zeroes; compare numeric values rather than the literal response text. `issues` is always an array and may be empty.

An issue contains `field` (a key or null) and `code`. Field-level issues for array
results also contain zero-based `index`. Record-level `not_found` or
`ambiguous_record` issues have `field=null`. Their data shape is described in
[record semantics](../explanation/records.md).

Successful extraction responses include:

| Header | Meaning |
|---|---|
| `Server-Timing` | Milliseconds for `queue`, `preprocess`, `inference`, `normalize`. |
| `X-Tractor-Input-Tokens` | Combined model input token count for schema and text. |

## Request-level errors

```json
{"error":{"code":"invalid_request","message":"Invalid extraction request"}}
```

| HTTP | Code | Meaning |
|---:|---|---|
| 413 | `body_limit` | Body exceeds 131,072 bytes. |
| 422 | `invalid_request` | Invalid JSON or request/schema validation failure. |
| 422 | `token_limit` | Joint schema/text exceeds the bundle's 512-token window. |
| 422 | `output_capacity` | Native record or per-field candidate capacity reached. |
| 503 | `not_ready` | Model unavailable for inference. |
| 503 | `overloaded` | Admission capacity occupied or inference wait timed out. |
| 500 | `internal_error` | Inference failed; model is marked unready. |

These are application errors for the documented endpoints; unknown paths/methods
may use the framework's own error format. There is no batch, streaming, authentication,
model-management, or OpenAI-compatible chat endpoint. Extraction issues are
separate from HTTP errors; see [the complete issue table](quantities-and-issues.md).

[Manual home](../index.md)
