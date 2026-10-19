import pytest

from tractor.corpus_expansion import product_episode


def product_row():
    return {'id': 'sample', 'source_group': 'layout-a', 'scenario': 'measurement', 'origin': 'synthetic',
            'metadata': {'language': 'de'},
            'request': {'text': 'Box: 12 kg', 'anchor': 'name',
                        'fields': {'name': {'type': 'text', 'description': 'Produktname'},
                                   'weight': {'type': 'quantity', 'description': 'Gewicht', 'unit': 'kg'}}},
            'gold': {'response': {'issues': []}, 'records': [
                {'name': [{'start': 0, 'end': 3, 'text': 'Box'}],
                 'weight': [{'start': 5, 'end': 10, 'text': '12 kg'}]}]}}


def test_replay_preserves_evidence_and_training_group():
    episode = product_episode(product_row())
    assert episode.split == 'train'
    assert episode.source_group == 'product_replay:layout-a'
    assert episode.tasks[0].kind == 'records'
    assert episode.tasks[0].id == 'record'
    assert episode.tasks[0].anchor == 'f0'
    assert episode.tasks[0].fields == {
        'f0': 'Produktname',
        'f1': 'Gewicht; vollständige Mengenangabe inklusive Einheit und Einschränkungen'}
    assert episode.gold['record'].records[0]['f1'].items[0].text == '12 kg'


def test_ambiguous_product_answers_are_not_relabelled_as_negatives():
    row = product_row()
    row['request']['text'] += ' or 13 kg'
    row['gold']['records'][0]['weight'].append({'start': 14, 'end': 19, 'text': '13 kg'})
    row['gold']['response']['issues'] = [{'field': 'weight', 'code': 'ambiguous_value'}]
    answer = product_episode(row).gold['record'].records[0]['f1']
    assert answer.status == 'present'
    assert [span.text for span in answer.items] == ['12 kg', '13 kg']


def test_normalization_failure_preserves_extractive_evidence():
    row = product_row()
    row['gold']['response']['issues'] = [{'field': 'weight', 'code': 'incompatible_dimension'}]
    assert product_episode(row).gold['record'].records[0]['f1'].items[0].text == '12 kg'


def test_replay_requires_known_complete_synthetic_annotation():
    row = product_row()
    del row['origin']
    with pytest.raises(ValueError, match='complete synthetic gold'):
        product_episode(row)


def test_replay_rejects_invented_offsets():
    row = product_row()
    row['gold']['records'][0]['weight'][0]['start'] = 0
    with pytest.raises(ValueError, match='original text'):
        product_episode(row)


def test_pool_screen_quarantines_lower_priority_groups(tmp_path):
    from tractor.corpus_expansion import screen_pool
    from tractor.jsonio import dumps, loads
    from tractor.source_import import qa

    text = 'Alice visits the quiet office beside the old railway station every morning before the daily planning meeting.'
    episode = qa({'id': 'sample', 'title': 'Sample', 'context': text,
                  'question': 'Who visits the office?',
                  'answers': {'text': ['Alice'], 'answer_start': [0]}}, 'squad_v2')
    for split in ('train', 'dev', 'calibration', 'test'):
        row = episode.model_copy(update={'id': split, 'source_group': split,
                                         'derivation_group': split, 'split': split})
        (tmp_path / f'{split}.episodes.jsonl').write_text(dumps(row.model_dump()) + '\n')
    screen_pool(tmp_path)
    assert loads((tmp_path / 'near-duplicate-blocked-groups.json').read_text()) == ['dev', 'train']
    assert loads((tmp_path / 'reserved-overlap-groups.json').read_text()) == ['calibration']
