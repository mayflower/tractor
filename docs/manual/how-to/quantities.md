# How to normalize quantities

Specify the unit your application needs, using the canonical spelling in the
[unit reference](../reference/quantities-and-issues.md). The source may use a supported
alias such as `inches`, `Zoll`, or `cm²`.

```python
from tractor import Client, Quantity

with Client() as client:
    area = client.extract("Die Fläche beträgt 12 cm².",
                          {"area_mm2": Quantity("Fläche", unit="mm2")})
    volume = client.extract("Das Volumen beträgt 1,5 L.",
                            {"volume_ml": Quantity("Volumen", unit="mL")}, locale="de-DE")
    speed = client.extract("Das Fahrzeug fährt mit 72 km/h.",
                           {"speed_ms": Quantity("Geschwindigkeit", unit="m/s")})
```

Expected values are `1200`, `1500`, and `20`, respectively. Target units are exact,
case-sensitive identifiers: use `mL`, `m3`, and `mm2` in requests.

## Set the source locale

Use `locale="de-DE"` for German separators and `en-US` or `en-GB` for English
separators. Locale describes numeric syntax, not the requested output language.
Without a locale, the parser tries German and US English interpretations and
accepts a value only when the successful interpretations agree. Thus `1,234 cm`
can require a locale; do not silently choose one based on which value seems plausible.

## Preserve numeric precision

The SDK returns JSON fractions as `Decimal`; the API sends JSON numbers. Preserve
them when writing another JSON response:

```python
from tractor.jsonio import dumps

payload = dumps({"data": volume.data, "issues": volume.issues})
```

This emits a numeric token, not a quoted decimal string. Python's standard
`json.dumps` cannot serialize Decimal by default. Avoid converting to `float` merely
to serialize. Conversion uses Decimal arithmetic, with 34 significant digits in
the final normalized value. Infinite decimal expansions are rounded; exact binary
floating-point arithmetic is not the normalization contract.

## Handle quantities that are not single values

A range, approximation, tolerance, bound, or chained dimension produces null with
`non_scalar_quantity` when selected as a quantity span. Fractions such as `1/2 inch`
are unsupported number notation. `gallon` and `Pfund` are ambiguous units.
Temperature conversion is outside the supported dimensions.

Do not change an unsupported temperature field to a length target to force a
result. Store unsupported information separately as text or route it for review.
A text field returns the original substring and does not normalize it.

[Manual home](../index.md)
