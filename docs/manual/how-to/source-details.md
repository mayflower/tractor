# How to highlight source evidence

Set `include_details=True` in the SDK or `include_details: true` in HTTP. For an
object result, details map field keys to source objects or null. For an array result,
details are an array in the same record order.

```python
from tractor import Client, Quantity

text = "📦 Atlas ist 12,5 cm breit."
with Client() as client:
    result = client.extract(text, {"width_mm": Quantity("Breite", unit="mm")},
                            locale="de-DE", include_details=True)
span = result.details["width_mm"]
if span is not None:
    highlight = text[span["start"]:span["end"]]
    assert highlight == span["text"]
```

Offsets address the exact submitted text in Unicode code points, starting at zero;
`end` is exclusive. An emoji occupies one Python code point but may occupy two
JavaScript UTF-16 code units. In JavaScript, slice the code-point array:

```javascript
const source = Array.from(text).slice(span.start, span.end).join("");
```

Render the slice as text, not unescaped HTML. For DOM highlighting, construct text
nodes or use your framework's normal text escaping. These offsets are not UTF-8
byte offsets and not grapheme-cluster indices. Combining characters remain separate
code points, so do not normalize Unicode after storing the original offsets.

For a chunk starting at code-point offset `chunk_start` in the original document,
add that offset to both returned positions. This mapping is valid only when the
chunk is an unchanged substring. If preprocessing removes headers or changes text,
maintain an explicit mapping instead.

A quantity's details can be present even when normalization produces an issue:
they explain which span was rejected. Missing or multiply ambiguous candidates can
have null details. `extraction_score`, when present, is a model score rather than
a calibrated probability that the final normalized answer is correct.

[Manual home](../index.md)
