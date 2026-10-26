from collections import Counter
from pathlib import Path

import pytest

from tractor.biqwen_attention import BIDIRECTIONAL
from tractor.biqwen_batching import batch_indices
from tractor.biqwen_training import FAST_RECIPE, epoch_order, update_batches, validate_config
from tractor.jsonio import loads


def fast_config():
    config = loads(Path('configs/biqwen-full.json').read_text())
    config.update(recipe=FAST_RECIPE, attention_mode=BIDIRECTIONAL, runtime={
        'gdn_backend': 'fla', 'max_microbatch': 8, 'padded_token_budget': 2048,
        'checkpointing_above_tokens': 512})
    return config


@pytest.mark.parametrize(('key', 'value'), [('max_steps', 128), ('sampling', True), ('quotas', True),
    ('validation_limit', 128), ('early_stopping', True)])
def test_acceleration_never_licenses_reduced_full_training(key, value):
    config = fast_config()
    validate_config(config)
    config[key] = value
    with pytest.raises(ValueError, match='forbids'):
        validate_config(config)


@pytest.mark.parametrize(('key', 'value'), [('max_microbatch', 0), ('padded_token_budget', True),
    ('checkpointing_above_tokens', -1), ('gdn_backend', 'automatic')])
def test_runtime_limits_are_explicit_and_valid(key, value):
    config = fast_config()
    config['runtime'][key] = value
    with pytest.raises(ValueError, match='runtime'):
        validate_config(config)


def test_adaptive_batches_keep_update_members_tail_and_resume():
    runtime = fast_config()['runtime']
    lengths = {i: [100, 4096, 200, 500][i % 4] for i in range(35)}
    order = epoch_order(35, 42, 0)
    seen = []
    for update in update_batches(order, 16, 1):
        original = [i for batch in update for i in batch]
        actual = list(batch_indices(sorted(original, key=lengths.get), lengths, runtime))
        assert Counter(i for batch in actual for i in batch) == Counter(original)
        for batch in actual:
            assert len(batch) <= runtime['max_microbatch']
            assert len(batch) == 1 or max(lengths[i] for i in batch) * len(batch) <= runtime['padded_token_budget']
        seen.extend(i for batch in actual for i in batch)
    assert Counter(seen) == Counter(range(35))
    remaining = [i for update in update_batches(order, 16, 1, cursor=32) for batch in update for i in batch]
    assert remaining == order[32:] and len(remaining) == 3
