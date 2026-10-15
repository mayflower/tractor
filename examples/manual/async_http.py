"""Async HTTP integration with bounded concurrency and Decimal-preserving JSON."""

import asyncio
import os

import httpx

from tractor.jsonio import dumps, loads


async def main():
    semaphore = asyncio.Semaphore(1)
    async with httpx.AsyncClient(
        base_url=os.environ.get("TRACTOR_URL", "http://127.0.0.1:8000"), timeout=10,
    ) as http:
        async def extract(text):
            payload = {"text": text, "locale": "de-DE", "fields": {
                "width_mm": {"type": "quantity", "description": "Breite", "unit": "mm"}}}
            async with semaphore:
                response = await http.post("/v1/extract", json=payload)
            response.raise_for_status()
            return loads(response.content)

        results = await asyncio.gather(
            extract("Breite: 12,5 cm."), extract("Modell CM125 hat keine Maßangaben."),
        )
        assert results[0]["data"]["width_mm"] == 125
        assert results[1]["data"]["width_mm"] is None
        for result in results:
            print(dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
