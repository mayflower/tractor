# Python SDK reference

The package requires Python `>=3.12,<3.13`. The public imports are:

```python
from tractor import (Address, Client, Date, Datetime, Duration, Email, Iban,
                     Identifier, Interval, List, Money, Number, Organization,
                     Percent, Person, Phone, Place, Postcode, Quantity, Result,
                     Text, Time, TractorError, Url)
```

## Installation

For SDK-only development in a separate checkout:

```sh
nix develop
just setup-sdk
```

This installs SDK dependencies and development tools without the model stack.
Do not run it in a serving environment you intend to keep fully installed: uv
sync may remove extras. Use `just setup` for a combined server/development checkout.
For an application environment with Python 3.12, install the local source package:

```sh
python -m pip install /path/to/tractor
```

This source-path command installs the base SDK dependencies. Do not assume a PyPI
package with the same name is this repository. A remote API client needs no model
weights, PyTorch, or GPU.

## Field classes

`Text(description: str)` declares a source-text field. `Quantity(description: str,
*, unit: str)` declares a quantity. These are frozen, strict Pydantic models.
Unknown constructor properties and unsupported units raise `ValidationError`.
Also available are `Date`, `Number`, `Money`, `Address`, `Identifier`, `List`,
`Person`, `Organization`, `Place`, `Phone`, `Email`, `Postcode`, `Time`,
`Datetime`, `Duration`, `Interval`, `Percent`, `Iban`, and `Url`.
Their [type-specific parameters](../how-to/document-processing.md) select processing
without a separate normalizer configuration. Descriptions are 1–240 characters. See [request constraints](http.md).

## Client

`Client(base_url="http://127.0.0.1:8000", timeout=5.0)` creates a synchronous reusable
httpx client. Use a context manager or call `close()`.

```text
client.extract(text, fields, *, locale=None, many=False, anchor=None)
```

`fields` contains typed field objects or equivalent field dictionaries.
The SDK validates the complete request before HTTP. There are no retry, streaming,
async, auth-header, or model-loading methods on this client. Use a custom HTTP client
for an authenticated gateway or async application.

## Result

| Attribute | Value |
|---|---|
| `data` | Dict or list of dicts; text, numeric, and null field values. |
| `issues` | List of issue dictionaries. |

JSON numbers with a decimal point or exponent decode to `decimal.Decimal`,
integer literals to `int`,
and JSON null to `None`. `Result` is a frozen dataclass; its nested dicts/lists are
not deep-immutable. Do not infer numeric type from a quantity field alone.

`tractor.jsonio.loads` and `tractor.jsonio.dumps` are available for Decimal-preserving
JSON, duplicate-key rejection, and finite-number validation. `dumps` returns a string;
encode it as UTF-8 for a byte-oriented transport.

## Exceptions

- `pydantic.ValidationError`: invalid local field/request definition.
- `TractorError`: HTTP 4xx/5xx response; attributes `status`, `code`, and the message via `str(exc)`.
- `httpx.TransportError` and subclasses: connection, timeout, or transport failure.
- JSON parsing/shape errors can propagate if an intermediary returns an unexpected successful body.

For non-JSON HTTP error bodies, the SDK falls back to `code="http_error"` when it
cannot decode the expected error envelope. It does not implement retry or repair.

[Manual home](../index.md)
