"""Small synchronous HTTP client."""

from dataclasses import dataclass
from typing import Any

import httpx

from .jsonio import dumps, loads
from .types import ExtractRequest, TractorError


@dataclass(frozen=True)
class Result:
    data: dict[str, Any] | list[dict[str, Any]]
    issues: list[dict[str, Any]]
    details: Any = None


class Client:
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=5.0):
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def extract(self, text, fields, *, locale=None, many=False, anchor=None, include_details=False):
        request = ExtractRequest(
            text=text,
            fields=fields,
            locale=locale,
            many=many,
            anchor=anchor,
            include_details=include_details,
        )
        response = self._http.post(
            "/v1/extract",
            content=dumps(request.model_dump()).encode(),
            headers={"Content-Type": "application/json"},
        )
        if response.is_error:
            try:
                error = loads(response.content)["error"]
            except (ValueError, KeyError, TypeError):
                error = {"code": "http_error", "message": response.reason_phrase}
            raise TractorError(response.status_code, error["code"], error["message"])
        return Result(**loads(response.content))

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
