# How to integrate HTTP and JavaScript clients

The public contract is HTTP JSON, so a custom client does not need Python or ML
libraries. Use `POST /v1/extract` with the same request schema as the SDK.

## curl

```sh
curl --fail-with-body --max-time 10 http://127.0.0.1:8000/v1/extract \
  -H 'Content-Type: application/json' --data-binary @examples/manual/records.json
```

Add `-i` to inspect `Server-Timing` and `X-Tractor-Input-Tokens`. Keep the application
ID outside the request object; unknown request properties are rejected.

## JavaScript with fetch

Use Node.js with built-in `fetch` and `AbortSignal.timeout`:

```sh
node examples/manual/node.mjs
```

The complete program sends the record fixture and prints the raw JSON response.
It checks `response.ok`; fetch itself does not throw for HTTP 422 or 503. Its
10-second timeout bounds the client's wait, but a disconnected client's inference
may continue on the server.

For ordinary JavaScript numbers, parse with `JSON.parse(body)`. For exact decimals,
keep the raw JSON during forwarding or use a decimal-aware JSON parser and decimal
arithmetic library that preserve number literals. Converting an already parsed
JavaScript Number to a decimal library cannot recover precision lost during parsing.

## Browsers and other services

The server does not install CORS middleware, authentication, or TLS. For a browser
application, call tractor through your own backend or an explicitly configured
same-origin gateway. Keep access control and public request limits at that boundary.
For Go, Java, PHP, or other clients, use the same endpoint and distinguish non-2xx
errors from successful responses with field-level issues. Configure the language's
JSON decoder to preserve decimals if your domain requires exact quantities.

There is no streaming response or server-side batch endpoint. To process multiple
inputs, make bounded individual requests or use the [batch example](../tutorials/batch.md).

[HTTP schema](../reference/http.md) · [Retry policy](errors-and-retries.md)

[Manual home](../index.md)
