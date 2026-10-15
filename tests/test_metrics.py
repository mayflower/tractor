from tractor.data import read_rows
from tractor.evaluate import metrics, score
from tractor.service import resolve
from tractor.types import ExtractRequest


def test_all_null_cannot_win():
    row = read_rows("tests/fixtures/api-cases.jsonl")[0]
    request = ExtractRequest.model_validate(row["request"])
    missing = [{name: [] for name in request.fields}]
    result = resolve(request, missing)
    m = metrics(score(row, missing, result))
    assert m["field_f1"] == 0
    assert m["coverage"] == 0
    assert m["counts"]["fn"] == len(request.fields)


def test_record_identity_uses_offsets():
    row = read_rows("tests/fixtures/api-cases.jsonl")[13]
    request = ExtractRequest.model_validate(row["request"])
    perfect = metrics(score(row, row["gold"]["records"], row["gold"]["response"]))
    assert perfect["field_f1"] == 1
    # Same product string at a different occurrence cannot stand in for its record.
    only_second = row["gold"]["records"][1:]
    partial = metrics(score(row, only_second, resolve(request, only_second)))
    assert partial["counts"]["fn"] == 2
    assert partial["counts"]["matched_records"] == 1
    assert partial["coverage"] == 0.5
