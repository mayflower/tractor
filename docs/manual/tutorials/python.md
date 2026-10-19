# Tutorial: build a Python product extractor

Start the server using [the first tutorial](first-extraction.md). You will extract
two product records and inspect their normalized values. Use a second terminal in the
checkout with `nix develop` active.

## 1. Declare your output fields

```python
from tractor import Client, Quantity, Text

fields = {
    "product": Text("Produktbezeichnung"),
    "width_mm": Quantity("Breite", unit="mm"),
}
```

Each field has a description. Quantity fields require a canonical target unit.
These declarations also validate locally before the SDK sends a request.

## 2. Extract two records

```python
with Client(timeout=10.0) as client:
    result = client.extract(
        "Alpha ist 10 cm breit. Beta ist 2 inches breit.",
        fields,
        anchor="product",
        many=True,
    )
    print(result.data)
    print(result.issues)
```

Expected data: `Alpha` with width `100`, followed by `Beta` with width `50.8`.
The fractional value is a Python `Decimal`. `anchor="product"` associates each
measurement with its product; `many=True` requests an array.

## 3. Run the complete example

```sh
uv run --no-sync python examples/manual/python_client.py
```

The program exercises records, decimal preservation, Unicode input, and a
missing measurement. It exits with an assertion error if the documented behavior
changes. These few examples check an integration, not real-world model accuracy.

Next: [batch import](batch.md), [SDK reference](../reference/python.md), or
[record behavior](../explanation/records.md).

[Manual home](../index.md)
