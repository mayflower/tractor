"""Compare a checkpoint and bundle from fresh processes with empty HF caches."""

import argparse
import os
import tempfile
from pathlib import Path

from tractor.data import read_rows
from tractor.jsonio import dumps, loads
from tractor.model import Extractor
from tractor.service import extract
from tractor.types import ExtractRequest

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--selection", help="Optional dev threshold selection")
args = parser.parse_args()
with tempfile.TemporaryDirectory(prefix="tractor-empty-hf-") as cache:
    os.environ["HF_HOME"] = cache
    extractor = Extractor(args.model, "cpu")
    if args.selection:
        extractor.thresholds = loads(Path(args.selection).read_text())["thresholds"]
    results = []
    for row in read_rows("tests/fixtures/api-cases.jsonl"):
        request = ExtractRequest.model_validate(row["request"]).model_copy(
            update={"include_details": True}
        )
        result = extract(extractor, request)
        details = result["details"] if isinstance(result["details"], list) else [result["details"]]
        for record in details:
            for source in record.values():
                if source:
                    source.pop("extraction_score", None)
        results.append({"id": row["id"], "result": result})
    Path(args.output).write_text(dumps(results, indent=2))
    assert not list(Path(cache).rglob("*")), "Unexpected HF cache dependency"
    print("Saved fresh-process outputs for", len(results), "requests; HF cache stayed empty")
