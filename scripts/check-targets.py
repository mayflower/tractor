"""Verify native token targets reconstruct every original gold occurrence."""

from pathlib import Path

from gliner2.training.trainer import ExtractorCollator

from tractor.data import read_rows
from tractor.jsonio import dumps
from tractor.model import Extractor, field_ids, native_example
from tractor.training_adapter import configure_processor
from tractor.types import ExtractRequest

Path("reports").mkdir(exist_ok=True)
x = Extractor("models/tractor", "cpu")
configure_processor(x.model.processor)
collate = ExtractorCollator(
    x.model.processor, is_training=True, architecture="boundary", max_gold_per_query=64
)
rows = (
    read_rows("tests/fixtures/api-cases.jsonl")
    + (read_rows("data/bootstrap/train.jsonl")[:96] if Path("data/bootstrap/train.jsonl").exists() else [])
)
for row in rows:
    request = ExtractRequest.model_validate(row["request"])
    native = native_example(row)
    batch = collate([(native["input"], native["output"])])
    inverse = {v: k for k, v in field_ids(request).items()}

    def chars(pair, batch=batch):
        start, end = map(int, pair)
        return (batch.start_mappings[0][start], batch.end_mappings[0][end - 1])

    if request.anchor:
        spans, mask, records, query_ids, *_ = batch.targets.record_targets
        assert int(records.sum()) == len(row["gold"]["records"]), row["id"]
        actual = []
        for ri in range(int(records.sum())):
            record = {}
            for fi, qi in enumerate(query_ids[0, 0]):
                name = inverse[batch.query_layouts[0].query(int(qi)).role_name]
                record[name] = [chars(pair) for pair in spans[0, 0, ri, fi][mask[0, 0, ri, fi]]]
            actual.append(record)
    else:
        actual = [{}]
        for query in batch.query_layouts[0].queries:
            qi = query.query_id
            pairs = batch.targets.mention_pairs[0, qi][batch.targets.mention_mask[0, qi]]
            actual[0][inverse[query.role_name]] = [chars(pair) for pair in pairs]
    expected = [
        {name: [(s["start"], s["end"]) for s in values] for name, values in record.items()}
        for record in row["gold"]["records"]
    ]
    assert actual == expected, (row["id"], actual, expected)
Path("reports/native-target-check.json").write_text(
    dumps({"samples": len(rows), "exact_original_offset_targets": True}, indent=2)
)
print("Native target reconstruction passed:", len(rows))
