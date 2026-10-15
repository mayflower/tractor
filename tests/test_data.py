from tractor.data import generate, read_rows, validate_rows
from tractor.service import resolve
from tractor.types import ExtractRequest


def test_generated_oracles(tmp_path):
    generate(tmp_path, sizes=(96, 48, 48))
    groups = []
    for split in ("train", "dev", "test"):
        rows = read_rows(tmp_path / f"{split}.jsonl")
        validate_rows(rows)
        groups.append({r["source_group"] for r in rows})
        for row in rows:
            actual = resolve(ExtractRequest.model_validate(row["request"]), row["gold"]["records"])
            expected = row["gold"]["response"]
            assert actual["issues"] == expected["issues"], row["id"]
            actual_rows = actual["data"] if isinstance(actual["data"], list) else [actual["data"]]
            expected_rows = (
                expected["data"] if isinstance(expected["data"], list) else [expected["data"]]
            )
            for a, e in zip(actual_rows, expected_rows, strict=True):
                for k, v in e.items():
                    if isinstance(v, (int, float)) or hasattr(v, "as_tuple"):
                        from decimal import Decimal

                        assert abs(a[k] - v) <= max(abs(v), Decimal(1)) * Decimal("1e-32"), row[
                            "id"
                        ]
                    else:
                        assert a[k] == v, row["id"]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
