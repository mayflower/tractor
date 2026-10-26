from collections import Counter

import pytest

from tractor.biqwen_feasibility import balanced_indices, microbatches


def test_diagnostic_includes_minority_language_and_records_without_duplicates():
    rows = [{'id': str(i), 'language': language, 'form': form, 'domain': 'contacts', 'source': 'test'}
            for i, (language, form) in enumerate(
                [('en', 'entities')] * 1000 + [('de', 'entities')] * 20 +
                [('en', 'records')] * 20 + [('de', 'records')] * 20)]
    selected = balanced_indices(rows, 32, 42)
    assert len(set(selected)) == 32
    assert Counter((rows[i]['language'], rows[i]['form']) for i in selected) == {
        ('en', 'entities'): 8, ('de', 'entities'): 8, ('en', 'records'): 8, ('de', 'records'): 8}
    assert selected == balanced_indices(rows, 32, 42)
    assert selected != balanced_indices(rows, 32, 43)
    with pytest.raises(ValueError, match='Insufficient'):
        balanced_indices(rows, len(rows) + 1, 42)


def test_length_batches_preserve_every_request_including_long_inputs_and_tail():
    lengths = [100, 100, 4096, 512, 1024, 300, 300]
    batches = list(microbatches(range(len(lengths)), lengths, max_batch=3, token_budget=1024))
    assert [i for batch in batches for i in batch] == list(range(len(lengths)))
    assert [2] in batches
    for batch in batches:
        assert len(batch) <= 3
        assert len(batch) == 1 or max(lengths[i] for i in batch) * len(batch) <= 1024


def test_diagnostic_rejects_out_of_scope_language():
    with pytest.raises(ValueError, match='DE/EN'):
        balanced_indices([{'id': 'cord', 'language': 'id', 'form': 'entities', 'domain': 'receipt', 'source': 'cord'}], 1, 42)
