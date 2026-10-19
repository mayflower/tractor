# Records, missing values, and ambiguity

An anchor establishes record occurrences. If `Atlas` appears twice with separate
measurements, those are two occurrences even though their names are equal. The
native record decoder associates each selected field with an anchor; the response
is sorted by the anchor's position in the original text.

| Request mode | Detected anchors | Data shape and record outcome |
|---|---:|---|
| No anchor, `many=false` | Not applicable | One object of document-level fields. |
| Anchor, `many=false` | 1 | One object. |
| Anchor, `many=false` | 0 | All-null object; `not_found` with `field=null`. |
| Anchor, `many=false` | More than 1 | All-null object; `ambiguous_record`. |
| Anchor, `many=true` | 1 or more | Array in source order. |
| Anchor, `many=true` | 0 | Empty array; `not_found` with `field=null`. |
| No anchor, `many=true` | Not applicable | Invalid request (422). |

A detected product with a missing width can remain a record with `width_mm=null`.
A product whose anchor is not detected cannot be represented as a returned record.
Therefore, an empty array does not prove the source contained no products.

Within a field, exact duplicate spans at the same offsets are deduplicated and
contained spans are removed in favor of maximal spans. Multiple independent
candidates produce `ambiguous_value`, even when their normalized numeric values
would be equal. Two separate `10 cm` occurrences are still two candidate sources.
This conservatism helps avoid silently making an association the system cannot
resolve under its contract.

Null expresses an unresolved value. Zero expresses a successfully normalized zero.
Issues explain the former. Applications can
choose different policies for optional attributes and required measurements, but
should keep that choice explicit and observable.

[Manual home](../index.md)

## Scope boundary

Record extraction groups values explicitly described together in readable text.
It does not reconstruct table rows, align separate columns, or recover invoice
positions from PDF/OCR layout. Use a separate table parser for those inputs.
Missing explicitly stated values in a text record remain extraction errors.
