# How to handle errors, retries, and backpressure

Distinguish three outcomes: an extracted value, a successful response with issues,
and a failed request. A 200 response can contain null values and issues. Handle
those as domain outcomes rather than retrying them indefinitely.

```python
import httpx
from pydantic import ValidationError
from tractor import Client, Quantity, TractorError

try:
    with Client(timeout=10.0) as client:
        result = client.extract("Breite: 12,5 cm.",
                                {"width_mm": Quantity("Breite", unit="mm")}, locale="de-DE")
except ValidationError:
    raise  # Fix the local field/request definition.
except TractorError as exc:
    print(exc.status, exc.code, exc)  # Server rejected or failed the request.
except httpx.TransportError as exc:
    print(type(exc).__name__)  # Timeout, connection, or other transport failure.
else:
    if result.issues:
        print("Review needed:", result.issues)
    else:
        print(result.data)
```

## Choose a bounded retry policy

A `422 invalid_request`, `token_limit`, or `output_capacity` needs a changed request,
not an identical retry. A `413 body_limit` needs a smaller body. A `500 internal_error`
marks the model unready; inspect and restart the service instead of repeatedly
hammering it. A `503 not_ready` should trigger a readiness check. A `503 overloaded`
can be retried with a small bounded backoff and jitter.

For example, wrap only a read-only extraction call:

```python
import random
import time
from tractor import TractorError

def with_backpressure(call):
    for attempt in range(3):
        try:
            return call()
        except TractorError as exc:
            if exc.code != "overloaded" or attempt == 2:
                raise
            time.sleep(0.25 * (2 ** attempt) + random.uniform(0, 0.1))
```

Use `with_backpressure(lambda: client.extract(text, fields))` with your existing
client. The SDK itself does not retry. Timeout retries may duplicate compute because
an abandoned request is shielded until inference finishes. Extraction has no database
write side effect, but your surrounding import or order workflow may have one;
make those writes idempotent by your own job ID.

## Route field issues

Keep `0` as a valid measurement and `None` as unresolved. For arrays, use the issue's
`index` to identify the affected record. A record-level issue can have `field=null`
and no index. If downstream logic requires every field, route the whole response
for review when `issues` is nonempty. If some fields are optional, apply an explicit
per-field policy rather than silently dropping all issues.

[Error reference](../reference/http.md) · [Issue codes](../reference/quantities-and-issues.md)

[Manual home](../index.md)
