from pathlib import Path

from gliner2.training.trainer import ExtractorCollator

from tractor.jsonio import dumps, loads
from tractor.model import Extractor, native_example
from tractor.training_adapter import configure_processor
from tractor.types import ExtractRequest, TractorError

Path("reports").mkdir(exist_ok=True)
x = Extractor("models/tractor", "cuda")
configure_processor(x.model.processor)
rows = [loads(s) for s in Path("tests/fixtures/api-cases.jsonl").read_text().splitlines()]
collator = ExtractorCollator(
    x.model.processor, is_training=True, architecture="boundary", max_gold_per_query=64
)
for row in rows:
    native = native_example(row)
    try:
        batch = collator([(native["input"], native["output"])])
        assert len(batch) == 1
    except Exception as exc:
        print(row["id"], type(exc).__name__, str(exc))
        raise
empty = {
    "request": {
        "text": "Keine Produkte vorhanden.",
        "fields": {"product": {"type": "text", "description": "Produkt"}},
        "anchor": "product",
        "many": True,
    },
    "gold": {"records": []},
}
n = native_example(empty)
b = collator([(n["input"], n["output"])])
print("ZERO RECORD", b.structure_labels, b.targets)
x.model.processor.change_mode(is_training=False)
results = []
for row in rows[:3] + [rows[13], rows[15]]:
    request = ExtractRequest.model_validate(row["request"])
    before = x.encoder_calls
    candidates = x.candidates(request)
    assert x.encoder_calls - before == 1
    results.append({"id": row["id"], "candidates": candidates})
for count in [1, 4, 8]:
    request = ExtractRequest(
        text="Die Breite ist 12 cm.",
        fields={
            f"w{i}": {"type": "quantity", "description": "Breite", "unit": "mm"}
            for i in range(count)
        },
    )
    before = x.encoder_calls
    x.candidates(request)
    assert x.encoder_calls - before == 1
request = ExtractRequest(
    text="lang " * 2000, fields={"p": {"type": "text", "description": "Produkt"}}
)
before = x.encoder_calls
try:
    x.candidates(request)
    raise AssertionError("Expected token rejection")
except TractorError as exc:
    assert exc.code == "token_limit"
assert x.encoder_calls == before
Path("reports/native-check.json").write_text(
    dumps(
        {"input_limit": x.input_limit, "fixture_collations": len(rows), "results": results},
        indent=2,
    )
)
print("Native collator, offsets, zero records and encoder counts OK")
