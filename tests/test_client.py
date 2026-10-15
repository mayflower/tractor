from decimal import Decimal

import httpx
import pytest

from tractor import Client, Quantity, TractorError
from tractor.jsonio import loads


def test_sdk_wire_and_no_retries():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            assert loads(request.content)["fields"]["width"]["unit"] == "mm"
            return httpx.Response(200, content=b'{"data":{"width":16.09344},"issues":[]}')
        return httpx.Response(503, json={"error": {"code": "overloaded", "message": "Busy"}})

    with Client() as client:
        client._http.close()
        client._http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
        result = client.extract("10 miles", {"width": Quantity("Breite", unit="mm")})
        assert result.data["width"] == Decimal("16.09344")
        with pytest.raises(TractorError) as error:
            client.extract("x", {"width": Quantity("Breite", unit="mm")})
        assert error.value.status == 503 and error.value.code == "overloaded"
        assert len(calls) == 2


def test_sdk_duplicate_response_rejected():
    with Client() as client:
        client._http.close()
        client._http = httpx.Client(
            base_url="http://test",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=b'{"data":{},"data":{},"issues":[]}')
            ),
        )
        with pytest.raises(ValueError):
            client.extract("x", {"width": Quantity("Breite", unit="mm")})
