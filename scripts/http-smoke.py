"""Real network/SDK smoke and encoder invariants; run with a local checkpoint."""

import argparse
import socket
import threading
import time
from decimal import Decimal

import httpx
import uvicorn

from tractor import Client, Quantity, Text
from tractor.jsonio import dumps
from tractor.model import Extractor
from tractor.server import create_app

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/tractor")
p.add_argument("--device", default="cpu")
a = p.parse_args()
x = Extractor(a.model, a.device)
app = create_app(extractor=x)
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
thread.start()
url = f"http://127.0.0.1:{port}"
try:
    for _ in range(300):
        if server.started:
            break
        if not thread.is_alive():
            raise RuntimeError("Server startup failed")
        time.sleep(0.1)
    with Client(url, timeout=30) as c:
        for n in (1, 4, 8):
            fields = {f"w{i}": Quantity("Breite", unit="mm") for i in range(n)}
            before = x.encoder_calls
            result = c.extract("Die Breite beträgt 12,5 cm.", fields, locale="de-DE")
            assert x.encoder_calls - before == 1
            print(dumps({"fields": n, "result": result.data, "issues": result.issues}))
        before = x.encoder_calls
        plain = c.extract(
            "📦 Atlas ist 12,5 cm breit.",
            {"p": Text("Produktbezeichnung"), "w": Quantity("Breite", unit="mm")},
            locale="de-DE",
        )
        assert x.encoder_calls - before == 1
        assert all(v is None or isinstance(v, (str, int, Decimal)) for v in plain.data.values())
    with httpx.Client(base_url=url, timeout=30) as c:
        before = x.encoder_calls
        assert c.post("/v1/extract", json={"text": "x", "fields": {}}).status_code == 422
        assert (
            c.post(
                "/v1/extract",
                json={
                    "text": "wort " * 2000,
                    "fields": {"p": {"type": "text", "description": "Produkt"}},
                },
            ).status_code
            == 422
        )
        assert x.encoder_calls == before
        assert c.get("/readyz").status_code == 200
        assert c.get("/openapi.json").status_code == 200
    print("Real HTTP, SDK, typed values, token rejection and encoder count checks passed.")
finally:
    server.should_exit = True
    thread.join(timeout=60)
    sock.close()
