# Extraction API

See the [user manual](manual/index.md) for tutorials and integration guides, and
the [complete HTTP reference](manual/reference/http.md) for the exact contract.

Start `tractor serve --model models/tractor --device cpu` and send JSON requests
to `POST /v1/extract`. The default address is `http://127.0.0.1:8000`.

| Request field | Meaning |
|---|---|
| `text` | Original source text |
| `fields` | Named field definitions with `type` and `description` |
| `locale` | Optional number-format locale, such as `de-DE` or `en-US` |
| `anchor` | Text-field key identifying records, such as `product` |
| `many` | Return an array of records when true |
| `include_details` | Include original source spans and offsets |

Field types are `text` and `quantity`. A quantity field also requires a target
`unit`. Without an anchor, extraction is document-wide. With an anchor, each
occurrence identifies a record, including repeated product names.

Responses contain `data` and `issues`, plus `details` when requested. Unresolved
fields are null. Issues identify the field and a code such as `not_found` or
`ambiguous_value`; array responses include the affected record index. Do not treat
null as zero. SDK fractional values use Decimal.

Malformed input, unsupported definitions and oversized requests return structured
errors. The joint schema/text limit is 512 tokens; requests are not silently
truncated. Admission is bounded; overload returns HTTP 503. Run one process per GPU.
Use the generated OpenAPI document at `/openapi.json` for the complete schema and
`/readyz` for readiness.

See [the SDK example](../examples/extract.py) for scalar fields, anchored records
and issue handling.
