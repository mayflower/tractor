# Units, numeric rules, and issue reference

## Canonical target units

| Dimension | Accepted target identifiers |
|---|---|
| Length | `mm`, `cm`, `m`, `km`, `in`, `ft`, `yd`, `mile`, `nmi` |
| Mass | `mg`, `g`, `kg`, `lb` |
| Time | `ms`, `s`, `min`, `h` |
| Volume | `mL`, `L`, `m3` |
| Area | `mm2`, `cm2`, `m2` |
| Speed | `m/s`, `km/h`, `mph` |

The identifiers are case sensitive. The source alias table additionally recognizes
forms such as `Zoll`, `inch`, `inches`, `lbs`, `Liter`, `litres`, `cm²`, and `m³`.
The exhaustive alias map is exported in the bundle's `normalization.json` and
implemented in [normalize.py](../../../src/tractor/normalize.py). Source aliases
are not automatically accepted as target identifiers.

## Numeric behavior

Supported locales: `de-DE`, `en-US`, `en-GB`. Without a locale, successful German
and US English parses must agree. Decimal/group separators are parsed strictly.
Space, nonbreaking-space, and narrow-nonbreaking-space grouping require complete
groups of three. Scientific notation and Unicode minus are supported. Fractions,
NaN, and Infinity are not accepted scalar values.

The parser rejects a mantissa digit string longer than 24 digits after stripping
leading zeroes, and exponents with absolute value above 100. Conversion uses a
50-digit Decimal context and rounds the final value to 34 significant digits with
round-half-even. Unit incompatibility is an issue, not an inferred conversion.

## Field and record issues

| Code | Meaning and usual action |
|---|---|
| `not_found` | No selected field span or no anchor record; review missing information. |
| `ambiguous_value` | Multiple independent candidates; resolve source ambiguity. |
| `ambiguous_record` | Single-record request detected multiple anchors; request many or narrow the input. |
| `invalid_span` | Candidate does not match its original source indices; treat as unresolved and investigate. |
| `non_scalar_quantity` | Range, tolerance, bound, approximation, or chained dimension. |
| `missing_unit` | Selected numeric value lacks an explicit unit. |
| `ambiguous_unit` | Unit such as gallon or Pfund has no unique supported meaning. |
| `unsupported_unit` | Source unit is outside the supported alias/dimension set. |
| `incompatible_dimension` | Supported source and target units have different dimensions. |
| `ambiguous_number_format` | Multiple successful locale interpretations disagree. |
| `invalid_number_format` | Separators/grouping or numeric literal are invalid. |
| `unsupported_number_format` | Unsupported notation, such as a fraction. |
| `numeric_out_of_range` | Numeric bounds exceeded or nonfinite conversion result. |

Issue selection depends on the actual selected source span. A model can miss a
qualifier or fail to select a span, so an illustrative source pattern does not
promise one particular issue under every wording. Normalization is deterministic
once the span and request settings are fixed.

For `many=true`, field issues have an `index`. Unresolved field values are null,
never a fabricated zero. See [HTTP errors](http.md) for request-level failures.

[Manual home](../index.md)
