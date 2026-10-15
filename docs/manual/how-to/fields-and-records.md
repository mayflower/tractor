# How to design fields and extract records

Use field keys for your downstream schema and descriptions for semantic intent.
A quantity target unit is mandatory and independent of the field's name:

```python
fields = {
    "product_width": Quantity("Breite des Produkts, nicht der Verpackung", unit="mm"),
    "package_width": Quantity("Breite der Verpackung", unit="mm"),
}
```

Import `Quantity` from `tractor`. On the text `Produkt Atlas ist 12 cm breit; die
Verpackung ist 20 cm breit.`, these fields distinguish the product's 120 mm width
from the packaging's 200 mm width. Short, specific descriptions reduce competition
between related properties. Evaluate wording on a development set; keep the final
test set separate.

## Choose document fields or records

For one description, omit `anchor` and `many`: `data` is an object. If several
products appear, declare a text field and use it as an anchor:

```python
from tractor import Client, Quantity, Text

with Client() as client:
    result = client.extract(
        "Atlas: Breite 10 cm. Atlas: Breite 12 cm.",
        {"product": Text("Produktbezeichnung"), "width_mm": Quantity("Breite", unit="mm")},
        anchor="product", many=True,
    )
```

This returns two records for the two occurrences of `Atlas`. An anchor is an
occurrence in the text, not a database entity key. Do not deduplicate by product
name unless your application deliberately performs entity resolution.

To require exactly one detected product, set an anchor and leave `many=False`.
Multiple detected anchors produce all-null data and `ambiguous_record`; no detected
anchor produces `not_found`. To request an array, set `many=True`; an anchor is then
required. No detected records produce an empty array plus `not_found`.

## Keep schemas and documents bounded

There are at most 32 fields, and descriptions have at most 240 characters. The
model's 512-token budget includes both schema and text, so wide schemas consume
input capacity. Avoid adding fields that the current workflow does not need.

Split long inputs at logical document or product boundaries before calling tractor.
Keep a record's name and measurements in the same chunk. Do not blindly cut a word
or measurement at a character limit; characters are not tokens. Retain each chunk's
start offset to map evidence back to the original document. If using overlapping
chunks, reconcile duplicate occurrences by original offsets rather than by name.

[HTTP reference](../reference/http.md) · [Records explained](../explanation/records.md)

[Manual home](../index.md)
