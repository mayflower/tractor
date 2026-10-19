import argparse
import itertools
import time
from pathlib import Path

from tractor.calibration import reuse_neural_outputs
from tractor.data import read_rows
from tractor.jsonio import dumps
from tractor.model import Extractor, schema_for
from tractor.types import ExtractRequest

parser = argparse.ArgumentParser(description="Real cached/direct product-policy parity check")
parser.add_argument("--model", default="models/tractor")
parser.add_argument("--data", default="tests/fixtures/api-cases.jsonl")
parser.add_argument("--limit", type=int, default=16)
args = parser.parse_args()
Path("reports").mkdir(exist_ok=True)
x = Extractor(args.model, "cpu")
grid = [
    dict(zip(("text", "quantity", "anchor"), values, strict=True))
    for values in itertools.product((0.25, 0.5, 0.75), repeat=3)
]
rows = read_rows(args.data)[:args.limit]
started = time.perf_counter()
for row in rows:
    request = ExtractRequest.model_validate(row["request"])
    ids = [
        x.model.processor.transform_record(request.text, schema_for(request, t)).input_ids
        for t in grid
    ]
    assert all(v == ids[0] for v in ids)
    hidden = x.encode(request)
    reference = []
    for thresholds in grid:
        x.thresholds = thresholds
        reference.append(x.candidates(request, hidden_states=hidden))
    with reuse_neural_outputs(x):
        for i, thresholds in enumerate(grid):
            x.thresholds = thresholds
            assert x.candidates(request, hidden_states=hidden) == reference[i], (
                row["id"],
                thresholds,
            )
report = {
    "samples": len(rows),
    "threshold_combinations": len(grid),
    "exact_match": True,
    "seconds": time.perf_counter() - started,
}
Path("reports/calibration-reuse-check.json").write_text(dumps(report, indent=2))
print(report)
