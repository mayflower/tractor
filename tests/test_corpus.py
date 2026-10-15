import random

import pytest

from tractor.corpus import SCENARIOS, generate, make_row
from tractor.corpus_audit import audit, check_oracle
from tractor.data import read_rows, validate_rows


@pytest.mark.parametrize("scenario", sorted(set(SCENARIOS)))
def test_reliability_independent_gold(scenario):
    # Each scenario spans both languages, all split syntaxes, dimensions and modes.
    rng = random.Random(183)
    for split in ("train", "dev", "test"):
        for language in ("de", "en"):
            for index in range(20):
                row = make_row(rng, split, index, language, index % 18, scenario)
                validate_rows([row])
                check_oracle(row)


def test_reproducible_split_safe_corpus(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(a, sizes=(200, 100, 100))
    generate(b, sizes=(200, 100, 100))
    for split in ("train", "dev", "test"):
        assert (a / f"{split}.jsonl").read_bytes() == (b / f"{split}.jsonl").read_bytes()
    report = audit(a)
    assert report["splits"]["test"]["invariance_pairs"] == 10
    assert report["cross_split_duplicate_texts"] == 0
    with pytest.raises(ValueError, match="never overwrite"):
        generate(a, sizes=(10, 10, 10))
    # The audit must fail on semantic corruption, not merely verify offsets.
    row = read_rows(a / "train.jsonl")[0]
    row["gold"]["response"]["issues"].append({"field": None, "code": "not_found"})
    with pytest.raises(ValueError, match="oracle mismatch"):
        check_oracle(row)


def test_translations_stay_in_same_split(tmp_path):
    manifest = generate(tmp_path, sizes=(10, 10, 10))
    groups = []
    for split in ("train", "dev", "test"):
        layouts = manifest["layout_split"][split]
        assert layouts["de"] == layouts["en"]
        groups.append(set(layouts["de"]))
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_audit_rejects_cross_split_family_leakage(tmp_path):
    from tractor.data import write_rows

    generate(tmp_path, sizes=(10, 10, 10))
    train = read_rows(tmp_path / "train.jsonl")
    dev = read_rows(tmp_path / "dev.jsonl")
    dev[0]["source_group"] = train[0]["source_group"]
    write_rows(tmp_path / "dev.jsonl", dev)
    with pytest.raises(ValueError, match="Cross-split leakage"):
        audit(tmp_path)
