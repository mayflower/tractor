"""Local Workers-AI-style REST route for the Clairic profile.

`account_id` is a local compatibility identifier, not a Cloudflare account.
The model alias always means the single local bundle loaded at startup; no
request can select a path or load weights. There is no Cloudflare
authentication: bind to loopback (the default) or put an authenticating proxy
in front. An Authorization header is accepted and ignored.
"""

import logging
import re

from fastapi import Request

from .capabilities import ClairicError, InvalidInput
from .clairic_compiler import Policy, compile_request
from .clairic_results import project
from .clairic_types import MODEL_ID, parse_request
from .jsonio import loads
from .service import extract_traced
from .types import TractorError

log = logging.getLogger(__name__)
ROUTE = "/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clairic"
ACCOUNT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Local TractorError codes mapped to local compatibility codes. These are
# never presented as Cloudflare error numbers.
NATIVE_CODES = {
    "body_limit": "tractor.capacity_exceeded",
    "token_limit": "tractor.capacity_exceeded",
    "output_capacity": "tractor.capacity_exceeded",
    "overloaded": "tractor.overloaded",
    "not_ready": "tractor.not_ready",
    "ambiguous_record": "tractor.unresolved_field",
    "internal_error": "tractor.internal_error",
}


def envelope(result=None, errors=()):
    return {"result": result, "success": not errors, "errors": list(errors), "messages": []}


def install_clairic(app, worker, body_limit, policy=None):
    from .server import JSONResponse, read_body

    policy = policy or Policy()

    def failure(status, error):
        return JSONResponse(envelope(errors=[error]), status_code=status)

    @app.post(ROUTE, response_class=JSONResponse)
    async def run(account_id: str, request: Request):
        try:
            if not ACCOUNT.fullmatch(account_id):
                raise InvalidInput("account_id must be a local identifier")
            raw = await read_body(request, body_limit)
            try:
                payload = loads(raw)
            except (ValueError, TypeError):
                raise InvalidInput("Request body is not valid JSON") from None
            parsed = parse_request(payload)
            if not app.state.ready:
                raise TractorError(503, "not_ready", "Model is not ready")
            # Compiled before admission: unsupported requests never reach the model.
            plan = compile_request(parsed, app.state.capabilities, policy)

            def function(extractor):
                result, trace = extract_traced(extractor, plan.extract_request)
                return project(plan, result, trace, uncertainty=app.state.uncertainty)

            payload, _ = await worker.submit(function, reject_if_busy=parsed.options.reject_if_busy)
            return JSONResponse(envelope(payload))
        except ClairicError as exc:
            return failure(exc.status, exc.describe())
        except TractorError as exc:
            return failure(exc.status, {"code": NATIVE_CODES.get(exc.code, "tractor." + exc.code),
                                        "message": str(exc)})

    @app.get("/v1/capabilities")
    async def capabilities():
        if not app.state.ready:
            raise TractorError(503, "not_ready", "Model is not ready")
        return {"model": MODEL_ID, "api": "clairic", "policy": vars(policy),
                "capabilities": app.state.capabilities.describe()}
