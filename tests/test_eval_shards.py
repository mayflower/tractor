from collections import Counter

import pytest

from tractor.eval_shards import merge
from tractor.evaluate import metrics


def part(counts):
    return [
        {
            "model": "model",
            "thresholds": {"text": 0.5},
            "metrics": metrics(Counter(counts)),
            "groups": {"document": metrics(Counter(counts))},
            "normalizer_on_gold": {"requests": counts["requests"]},
            "predictions": "raw.jsonl",
        }
    ]


def test_shards_sum_counts_not_rates():
    a = dict(requests=1, tp=1, fp=0, fn=0, answered=1, answerable=1)
    b = dict(requests=9, tp=0, fp=9, fn=9, answered=9, answerable=9)
    result = merge([part(a), part(b)], 10)[0]
    assert result["metrics"]["precision"] == 0.1
    assert result["metrics"] == metrics(Counter(a) + Counter(b))
    assert result["groups"]["document"]["precision"] == 0.1
    with pytest.raises(ValueError, match="Incomplete"):
        merge([part(a), part(b)], 11)
    with pytest.raises(ValueError, match="Incomplete"):
        merge([part(a), []], 1)
