"""Bounded single-model ASGI service."""

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


def create_app(
    model_path=None,
    device="cpu",
    precision="fp32",
    *,
    extractor=None,
    max_pending=1,
    wait_timeout=1.0,
    body_limit=128 * 1024,
):
    @asynccontextmanager
    async def lifespan(app):
        from .address_normalization import parsed_parts

        # Address parsing is part of the public contract. Load its local model
        # before readiness rather than failing on the first address request.
        await asyncio.to_thread(parsed_parts, 'London', 'GB')
        if extractor is not None:
            app.state.extractor = extractor
        else:
            from .model import Extractor

            app.state.extractor = await asyncio.to_thread(Extractor, model_path, device, precision)
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

    async def run(request):
        acquired = False
        started = time.perf_counter()
        try:
            try:
                await asyncio.wait_for(app.state.lock.acquire(), timeout=wait_timeout)
                acquired = True
            except TimeoutError:
                raise TractorError(503, "overloaded", "Inference wait timeout") from None
            # This job is shielded from client cancellation until the worker really completes.
            queue_ms = (time.perf_counter() - started) * 1000
            result = await asyncio.to_thread(extract, app.state.extractor, request)
            timing = {"queue": queue_ms, **getattr(app.state.extractor, "last_timings", {})}
            headers = {
                "Server-Timing": ", ".join(f"{k};dur={v:.3f}" for k, v in timing.items()),
                "X-Tractor-Input-Tokens": str(getattr(app.state.extractor, "last_tokens", 0)),
            }
            return result, headers
        finally:
            if acquired:
                app.state.lock.release()
            app.state.accepted -= 1

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
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > body_limit:
                raise TractorError(413, "body_limit", "Request exceeds body limit")
            raw.extend(chunk)
        try:
            parsed = ExtractRequest.model_validate(loads(bytes(raw)))
        except (ValueError, TypeError, ValidationError):
            raise TractorError(422, "invalid_request", "Invalid extraction request") from None
        if not app.state.ready:
            raise TractorError(503, "not_ready", "Model is not ready")
        if app.state.accepted >= 1 + max_pending:
            raise TractorError(503, "overloaded", "Inference capacity occupied")
        app.state.accepted += 1
        job = asyncio.create_task(run(parsed))
        app.state.jobs.add(job)

        def finished(task):
            app.state.jobs.discard(task)
            if not task.cancelled():
                task.exception()  # Consume errors even after client disconnect.

        job.add_done_callback(finished)
        try:
            result, headers = await asyncio.shield(job)
            return JSONResponse(result, headers=headers)
        except TractorError:
            raise
        except Exception:
            # Native errors may embed source text; do not log their exception strings.
            log.error("Inference failed; model marked unready")
            app.state.ready = False
            raise TractorError(500, "internal_error", "Inference failed") from None

    return app
