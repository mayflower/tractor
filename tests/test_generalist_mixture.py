import random
from collections import Counter

import pytest

from tractor.generalist_mixture import balanced, bucket


def test_balancing_limits_large_source_and_never_repeats_rows():
    rows = [({'source': source, 'family': 'named_entities', 'language': 'en', 'id': f'{source}:{i}'}, '', '')
            for source, count in [('large', 100), ('small', 10)] for i in range(count)]
    selected = balanced(rows, 20, random.Random(42))
    assert Counter(row[0]['source'] for row in selected) == {'large': 10, 'small': 10}
    assert len({row[0]['id'] for row in selected}) == 20
    with pytest.raises(ValueError, match='Insufficient'):
        balanced(rows, 111, random.Random(42))


def test_task_bucket_mapping_keeps_classification_together():
    assert bucket({'family': 'classification_routing'}) == 'classification'
    assert bucket({'family': 'classification_intent'}) == 'classification'
    assert bucket({'family': 'multiple_answers'}) == 'questions'
    assert bucket({'family': 'structured_records'}) == 'records'
