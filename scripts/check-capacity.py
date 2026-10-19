"""Check actual joint-context rejection boundaries without a latency benchmark."""

from pathlib import Path

from tractor.episodes import Task, compile_schema
from tractor.jsonio import dumps
from tractor.model import Extractor
from tractor.types import TractorError

x = Extractor("models/tractor", "cpu")
tasks = [Task(id="people", kind="entities", description="People", fields={"person": "Person name"})]
schema = compile_schema(tasks)
lo, hi = 1, 1024
while lo + 1 < hi:
    mid = (lo + hi) // 2
    tokens = len(x.model.processor.transform_record("word " * mid, schema).input_ids)
    if tokens <= x.input_limit:
        lo = mid
    else:
        hi = mid
report = {"joint_token_limit": x.input_limit, "cases": []}
for n in (lo - 1, lo, hi):
    text = "word " * n
    tokens = len(x.model.processor.transform_record(text, schema).input_ids)
    before = x.encoder_calls
    try:
        x.extract_tasks(text, tasks)
        status = "supported"
        assert tokens <= x.input_limit and x.encoder_calls - before == 1
    except TractorError as exc:
        status = exc.code
        assert tokens > x.input_limit and x.encoder_calls == before
    report["cases"].append({"tokens": tokens, "status": status,
                            "encoder_calls": x.encoder_calls - before})
Path("reports/training").mkdir(parents=True, exist_ok=True)
Path("reports/training/capacity.json").write_text(dumps(report, indent=2))
print(dumps(report, indent=2))
