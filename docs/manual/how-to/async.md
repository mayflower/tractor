# How to use async Python and web applications

The bundled `Client` is synchronous. In an async web route, either run it in a
worker thread or use `httpx.AsyncClient` directly. The latter keeps connection
pooling and cancellation in your application's async lifecycle.

```sh
uv run --no-sync python examples/manual/async_http.py
```

The [complete program](../../../examples/manual/async_http.py) creates one reusable
HTTP client, schedules two inputs, and uses a semaphore of one to bound requests
to a single tractor instance. It parses responses with `tractor.jsonio.loads` to
preserve Decimal values. HTTP errors raise `httpx.HTTPStatusError`; this raw HTTP
integration does not produce the SDK's `TractorError`.

## Integrate with an async request handler

Put the HTTP client and semaphore in your application's lifespan, create them once
per application worker, and close the client during shutdown. The core call is:

```python
async with semaphore:
    response = await http.post("/v1/extract", json=request_payload)
response.raise_for_status()
result = loads(response.content)
```

Here `http` is a configured `httpx.AsyncClient`, `semaphore` is an
`asyncio.Semaphore(1)`, and `loads` comes from `tractor.jsonio`. See the complete file
for imports and setup. When returning data through a framework, serialize Decimal
using `tractor.jsonio.dumps` into a JSON response body; default encoders may convert
it to binary floats or strings.

An application-wide queue is needed when several web workers share one tractor
server: a semaphore in each process does not enforce a global concurrency limit.
The server permits one active inference and one waiting request by default. For
higher throughput, add independent server instances behind a dispatcher and measure
your actual workload. A single GPU should run one serving process under this design.

Treat `503 overloaded` as backpressure. Increasing client concurrency can increase
errors and waiting time without improving model throughput. A client timeout does
not imply that the server stopped its work. See [retry handling](errors-and-retries.md).

[Manual home](../index.md)
