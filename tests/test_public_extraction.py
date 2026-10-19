import pytest

from tractor.public_extraction import ProtectedIndex, few_nerd, require_extraction
from tractor.source_import import qa


def test_few_nerd_preserves_adjacent_fine_boundaries_and_repeated_mentions():
    e = few_nerd({'id': 'x', 'tokens': ['Anna', 'Bob', 'met', 'Anna'],
                  'ner_tags': [7, 7, 0, 7], 'fine_ner_tags': [50, 51, 0, 50]})
    assert e.text == 'Anna Bob met Anna'
    assert e.split == 'train'
    assert e.original_split == 'train'
    assert [(s.start, s.end, s.text) for s in e.gold['entities'].records[0]['person'].items] == [
        (0, 4, 'Anna'), (5, 8, 'Bob'), (13, 17, 'Anna')]
    assert e.gold['entities'].records[0]['location'].status == 'absent'
    require_extraction(e)


def test_explicit_outside_tags_are_a_negative_not_invented_entities():
    e = few_nerd({'id': 'x', 'tokens': ['No', 'name'], 'ner_tags': [0, 0], 'fine_ner_tags': [0, 0]})
    assert all(a.status == 'absent' for a in e.gold['entities'].records[0].values())
    require_extraction(e)


@pytest.mark.parametrize('tags,fine', [([9], [1]), ([7], [0]), ([0], [1]), ([0, 0], [0])])
def test_invalid_annotation_is_rejected(tags, fine):
    with pytest.raises(ValueError):
        few_nerd({'id': 'x', 'tokens': ['Anna'], 'ner_tags': tags, 'fine_ner_tags': fine})


def test_synthetic_and_incomplete_targets_do_not_count():
    e = few_nerd({'id': 'x', 'tokens': ['Anna'], 'ner_tags': [7], 'fine_ner_tags': [50]})
    with pytest.raises(ValueError, match='real public'):
        require_extraction(e.model_copy(update={'text_origin': 'synthetic'}))
    e.gold['entities'].complete = False
    with pytest.raises(ValueError, match='incomplete'):
        require_extraction(e)


def test_classification_does_not_count_as_extraction():
    e = few_nerd({'id': 'x', 'tokens': ['Anna'], 'ner_tags': [7], 'fine_ner_tags': [50]})
    # Guard must reject the task kind even before inspecting its answers.
    e.tasks[0] = e.tasks[0].model_copy(update={'kind': 'classification'})
    with pytest.raises(ValueError, match='Out-of-scope'):
        require_extraction(e)


def test_protected_index_finds_sentence_inside_longer_passage():
    def example(text, title):
        return qa({'id': title, 'title': title, 'context': text, 'question': 'Who?',
                   'answers': {'text': [], 'answer_start': []}}, 'squad_v2')
    protected = ProtectedIndex()
    protected.add(example('Before. Alice visited the quiet museum beside the old railway station. Afterwards.', 'A'))
    candidate = example('Alice visited the quiet museum beside the old railway station .', 'B')
    assert protected.overlaps(candidate) == 'protected_lexical_containment'
    assert protected.overlaps(example('A completely unrelated short text.', 'C')) is None
    assert protected.overlaps(example('Another passage in the same source.', 'A')) == 'protected_source_group'
