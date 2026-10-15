"""Check the manual's published requests against a running release API (no paid APIs)."""

import os
from pathlib import Path

import httpx

from tractor.compare import exact
from tractor.jsonio import loads

cases = Path(__file__).with_name("cases.jsonl")
with httpx.Client(
    base_url=os.environ.get("TRACTOR_URL", "http://127.0.0.1:8000"), timeout=30,
) as client:
    for line in cases.read_text().splitlines():
        case = loads(line)
        response = client.post("/v1/extract", json=case["request"])
        response.raise_for_status()
        result = loads(response.content)
        assert exact({k: result[k] for k in ("data", "issues")}, case["expected"]), case["id"]
        if "details" in result:
            records = result["details"] if isinstance(result["details"], list) else [result["details"]]
            for record in records:
                for span in record.values():
                    if span is not None:
                        assert case["request"]["text"][span["start"]:span["end"]] == span["text"]
        print(case["id"], "passed")
print("All published integration cases passed; this is not an accuracy benchmark.")
