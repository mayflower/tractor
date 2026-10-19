import pytest

from tractor.synthetic_coverage import make_episode
from tractor.synthetic_paraphrases import mark, reconstruct


def test_roundtrip_preserves_offsets_and_train_group():
    original, _ = make_episode('train', 12, 'records', 'de')
    marked, _ = mark(original)
    revised = reconstruct(original, {'id': original.id, 'marked_text': 'Hallo!\n' + marked})
    assert revised.source_group == original.source_group
    assert revised.annotation == 'silver'
    assert revised.label_origin == 'teacher'
    for before, after in zip(original.gold['extract'].records, revised.gold['extract'].records, strict=True):
        for key in before:
            assert after[key].items[0].start == before[key].items[0].start + len('Hallo!\n')


def test_teacher_cannot_change_drop_or_duplicate_marked_fact():
    original, _ = make_episode('train', 12, 'relations', 'en')
    marked, mapping = mark(original)
    value = mapping[0]['text']
    for changed in [marked.replace(value, 'altered', 1), marked.replace('<v0>', '<v999>'),
                    marked + f'<v0>{value}</v0>', marked + ' ' + value]:
        with pytest.raises(ValueError):
            reconstruct(original, {'id': original.id, 'marked_text': changed})


def test_no_teacher_inputs_from_evaluation():
    episode, _ = make_episode('test', 12, 'records', 'de')
    with pytest.raises(ValueError):
        mark(episode)
