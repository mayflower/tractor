import asyncio
import threading

import httpx

from tractor.server import create_app


class BlockingExtractor:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def candidates(self, request):
        self.calls.append(request.text)
        if request.text == "block":
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError("Test deadline")
        if request.text == "fail":
            raise RuntimeError("private source text must not be logged")
        return [] if request.anchor else [{name: [] for name in request.fields}]


def body(text):
    return {"text": text, "fields": {"p": {"type": "text", "description": "Produkt"}}}


def test_disconnect_does_not_release_inference_lock():
    async def check():
        extractor = BlockingExtractor()
        app = create_app(extractor=extractor, wait_timeout=2)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(client.post("/v1/extract", json=body("block")))
                assert await asyncio.to_thread(extractor.entered.wait, 2)
                first.cancel()
                try:
                    await first
                except asyncio.CancelledError:
                    pass
                second = asyncio.create_task(client.post("/v1/extract", json=body("second")))
                await asyncio.sleep(0.05)
                assert "second" not in extractor.calls
                assert (await client.get("/healthz")).status_code == 200
                assert (await client.post("/v1/extract", json=body("third"))).status_code == 503
                extractor.release.set()
                assert (await second).status_code == 200
                assert "second" in extractor.calls
                assert (await client.post("/v1/extract", json=body("fail"))).status_code == 500
                assert (await client.get("/readyz")).status_code == 503

    asyncio.run(check())


def test_wait_timeout():
    async def check():
        extractor = BlockingExtractor()
        app = create_app(extractor=extractor, wait_timeout=0.05)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(client.post("/v1/extract", json=body("block")))
                assert await asyncio.to_thread(extractor.entered.wait, 2)
                response = await client.post("/v1/extract", json=body("second"))
                assert response.status_code == 503
                assert "second" not in extractor.calls
                extractor.release.set()
                assert (await first).status_code == 200

    asyncio.run(check())
