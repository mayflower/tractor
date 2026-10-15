"""Sequential JSONL import; preserve IDs, decimals, and per-item failures."""

import argparse
import os

import httpx

from tractor import Client, Quantity, TractorError
from tractor.jsonio import dumps, loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output", help="New JSONL file; existing files are not overwritten")
    args = parser.parse_args()
    fields = {"width_mm": Quantity("Breite", unit="mm")}
    url = os.environ.get("TRACTOR_URL", "http://127.0.0.1:8000")
    with (
        open(args.input, encoding="utf-8") as source,
        open(args.output, "x", encoding="utf-8") as target,
        Client(url, timeout=10) as client,
    ):
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item_id = f"line-{number}"
            try:
                item = loads(line)
                if not isinstance(item, dict):
                    raise ValueError("Expected one object per line")
                item_id = item.get("id", item_id)
                result = client.extract(item["text"], fields, locale=item.get("locale"))
                output = {"id": item_id,
                          "status": "needs_review" if result.issues else "extracted",
                          "data": result.data, "issues": result.issues}
            except TractorError as exc:
                output = {"id": item_id, "status": "error",
                          "error": {"http_status": exc.status, "code": exc.code}}
            except (ValueError, KeyError, TypeError, httpx.TransportError) as exc:
                output = {"id": item_id, "status": "error",
                          "error": {"code": type(exc).__name__}}
            target.write(dumps(output)+"\n")
            target.flush()


if __name__ == "__main__":
    main()
