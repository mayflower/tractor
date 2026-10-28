"""Bounded single-model ASGI service with one explicitly chosen API profile.

Both profiles share one model instance and one serialized inference worker:
at most one active job, a bounded queue, a wait timeout, and a lock held until
the worker thread really finishes, even after a client disconnect.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import ValidationError

from .jsonio import dumps, loads
from .service import extract
from .types import ExtractRequest, TractorError

log = logging.getLogger(__name__)


class JSONResponse(Response):
    media_type = "application/json"

    def render(self, content):
        return dumps(content).encode("utf-8")


API_PROFILES = ("tractor", "clairic")


class InferenceWorker:
    """Admission control and serialized execution shared by every profile."""

    def __init__(self, app, max_pending, wait_timeout):
        self.app = app
        self.max_pending = max_pending
        self.wait_timeout = wait_timeout

    def admit(self):
        state = self.app.state
        if not state.ready:
            raise TractorError(503, "not_ready", "Model is not ready")
        if state.accepted >= 1 + self.max_pending:
            raise TractorError(503, "overloaded", "Inference capacity occupied")
        state.accepted += 1

    async def _run(self, function, wait_timeout):
        state = self.app.state
        acquired = False
        started = time.perf_counter()
        try:
            try:
                await asyncio.wait_for(state.lock.acquire(), timeout=wait_timeout)
                acquired = True
            except TimeoutError:
                raise TractorError(503, "overloaded", "Inference wait timeout") from None
            # This job is shielded from client cancellation until the worker really completes.
            queue_ms = (time.perf_counter() - started) * 1000
            result = await asyncio.to_thread(function, state.extractor)
            return result, queue_ms
        finally:
            if acquired:
                state.lock.release()
            state.accepted -= 1

    async def submit(self, function, *, reject_if_busy=False):
        """Run function(extractor) once; errors other than TractorError mark the model unready."""
        self.admit()
        if reject_if_busy and self.app.state.lock.locked():
            self.app.state.accepted -= 1
            raise TractorError(503, "overloaded", "Inference capacity occupied")
        job = asyncio.create_task(self._run(function, self.wait_timeout))
        self.app.state.jobs.add(job)

        def finished(task):
            self.app.state.jobs.discard(task)
            if not task.cancelled():
                task.exception()  # Consume errors even after client disconnect.

        job.add_done_callback(finished)
        try:
            return await asyncio.shield(job)
        except TractorError:
            raise
        except Exception as exc:
            if getattr(exc, "keeps_ready", False):
                raise
            # Native errors may embed source text; do not log their exception strings.
            log.error("Inference failed; model marked unready")
            self.app.state.ready = False
            raise TractorError(500, "internal_error", "Inference failed") from None


async def read_body(request, body_limit):
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > body_limit:
            raise TractorError(413, "body_limit", "Request exceeds body limit")
        raw.extend(chunk)
    return bytes(raw)


def create_app(
    model_path=None,
    device="cpu",
    precision="fp32",
    *,
    extractor=None,
    max_pending=1,
    wait_timeout=1.0,
    body_limit=None,
    api="tractor",
    policy=None,
    uncertainty=None,
):
    if api not in API_PROFILES:
        raise ValueError(f"Unknown API profile: {api}")
    if body_limit is None:
        body_limit = 128 * 1024 if api == "tractor" else 1024 * 1024
    @asynccontextmanager
    async def lifespan(app):
        from .address_normalization import parsed_parts

        # Address parsing is part of the public contract. Load its local model
        # before readiness rather than failing on the first address request.
        await asyncio.to_thread(parsed_parts, 'London', 'GB')
        if extractor is not None:
            app.state.extractor = extractor
        else:
            from .backends import load_extractor

            app.state.extractor = await asyncio.to_thread(load_extractor, model_path, device, precision)
        warm = ExtractRequest(
            text="Atlas: Breite 12 cm.",
            fields={
                "product": {"type": "text", "description": "Produkt"},
                "width": {"type": "quantity", "description": "Breite", "unit": "mm"},
            },
        )
        await asyncio.to_thread(extract, app.state.extractor, warm)
        await asyncio.to_thread(
            extract,
            app.state.extractor,
            warm.model_copy(update={"anchor": "product", "many": True}),
        )
        if api == "clairic":
            from .capabilities import extractor_capabilities

            app.state.capabilities = extractor_capabilities(
                app.state.extractor, compute=bool(policy and policy.compute))
            provider = uncertainty(app.state.extractor) if callable(uncertainty) else uncertainty
            app.state.uncertainty = provider
            if provider is not None:
                for capability in provider.capabilities:
                    app.state.capabilities.add(capability)
        app.state.ready = True
        yield
        app.state.ready = False
        if app.state.jobs:
            await asyncio.gather(*app.state.jobs, return_exceptions=True)

    app = FastAPI(title="tractor", lifespan=lifespan)
    app.state.ready = False
    app.state.accepted = 0
    app.state.lock = asyncio.Lock()
    app.state.jobs = set()
    app.state.api = api
    worker = InferenceWorker(app, max_pending, wait_timeout)

    @app.exception_handler(TractorError)
    async def domain_error(request, exc):
        return JSONResponse(
            {"error": {"code": exc.code, "message": str(exc)}}, status_code=exc.status
        )

    @app.get("/healthz")
    async def health():
        return {"status": "alive"}

    @app.get("/readyz")
    async def ready():
        if not app.state.ready:
            raise TractorError(503, "not_ready", "Model is not ready")
        return {"status": "ready"}

    def native(request):
        def function(extractor):
            result = extract(extractor, request)
            timing = getattr(extractor, "last_timings", {})
            return result, dict(timing), getattr(extractor, "last_tokens", 0)
        return function

    if api == "clairic":
        from .clairic_server import install_clairic

        install_clairic(app, worker, body_limit, policy)
        return app

    schema = ExtractRequest.model_json_schema()
    schema["$defs"] = schema.get("$defs", {})

    # Inline definitions for a standalone OpenAPI request schema.
    def inline(value, definitions=None):
        definitions = schema["$defs"] if definitions is None else definitions
        if isinstance(value, dict):
            if "$ref" in value:
                return inline(definitions[value["$ref"].split("/")[-1]], definitions)
            return {k: inline(v, definitions) for k, v in value.items() if k not in {"$defs", "discriminator"}}
        if isinstance(value, list):
            return [inline(v, definitions) for v in value]
        return value

    scalar = {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "null"}]}
    record = {"type": "object", "additionalProperties": {"anyOf": [scalar, {"type": "array", "items": scalar}]}}

    @app.post(
        "/v1/extract",
        response_class=JSONResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": inline(schema)}},
            },
            "responses": {
                "200": {
                    "description": "Extraction result",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["data", "issues"],
                                "properties": {
                                    "data": {"anyOf": [record, {"type": "array", "items": record}]},
                                    "issues": {"type": "array", "items": {"type": "object"}},
                                },
                            }
                        }
                    },
                }
            },
        },
    )
    async def endpoint(request: Request):
        raw = await read_body(request, body_limit)
        try:
            parsed = ExtractRequest.model_validate(loads(raw))
        except (ValueError, TypeError, ValidationError):
            raise TractorError(422, "invalid_request", "Invalid extraction request") from None
        (result, timing, tokens), queue_ms = await worker.submit(native(parsed))
        timing = {"queue": queue_ms, **timing}
        headers = {
            "Server-Timing": ", ".join(f"{k};dur={v:.3f}" for k, v in timing.items()),
            "X-Tractor-Input-Tokens": str(tokens),
        }
        return JSONResponse(result, headers=headers)

    return app
