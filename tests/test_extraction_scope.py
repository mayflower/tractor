import json

import pytest

from tractor.coverage_assembly import sha
from tractor.extraction_scope import in_scope, prepare


@pytest.mark.parametrize('family', ['question_answering', 'multiple_answers'])
def test_qa_is_excluded_independent_of_source(family):
    assert not in_scope({'family': family, 'source': 'new_source'})


def test_qa_source_cannot_be_relabelled_as_document_fields():
    assert not in_scope({'family': 'document_fields', 'source': 'adversarial_qa'})
    assert in_scope({'family': 'document_fields', 'source': 'text_json'})


def test_filter_preserves_pairs_and_splits(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    for split in ('train', 'dev', 'calibration', 'test'):
        rows = [{'id': f'{split}:{family}', 'split': split, 'family': family, 'source': 'example',
                 'text': family, 'text_origin': 'real', 'label_origin': 'human'}
                for family in ['question_answering', 'document_fields']]
        (source / f'{split}.episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (source / f'{split}.native.jsonl').write_text(''.join(json.dumps({'input': r['text']})+'\n' for r in rows))
    (source / 'manifest.json').write_text(json.dumps({'status': 'complete', 'files':
        {p.name: sha(p) for p in source.glob('*.jsonl')}}))
    result = prepare(source, output)
    assert result['counts'] == dict.fromkeys(('train', 'dev', 'calibration', 'test'), 1)
    for split in result['counts']:
        assert json.loads((output / f'{split}.native.jsonl').read_text()) == {'input': 'document_fields'}


@pytest.mark.parametrize('family', ['classification_intent', 'classification_routing',
                                   'classification_urgency', 'classification_sentiment'])
def test_unwanted_classification_is_excluded_including_mixed_episodes(family):
    assert not in_scope({'family': family, 'source': 'synthetic',
                         'tasks': [{'kind': 'classification'}, {'kind': 'entities'}]})


def test_massive_is_excluded_independent_of_family():
    assert not in_scope({'family': 'document_fields', 'source': 'massive'})


def test_configured_classification_and_gates_only_keep_allowed_tasks():
    from pathlib import Path

    config = json.loads(Path('configs/classification-coverage.json').read_text())
    assert config['retired'] and not config['axes']
    assert not config.get('source')
    policy = json.loads(Path('configs/generalist-quality.json').read_text())
    assert policy['required_classification_tasks'] == []


def test_public_positions_excluded_but_header_preserved():
    from tractor.document_text_scope import restrict
    row = {'id': 'r', 'split': 'dev', 'source': 'cord', 'family': 'invoices',
           'artifact': 'repeated_records', 'text': 'TOTAL 20',
           'tasks': [{'id': 'head', 'kind': 'entities'}, {'id': 'items', 'kind': 'records'}],
           'gold': {'head': {}, 'items': {}}}
    assert not in_scope(row)
    kept, audit = restrict(row)
    assert kept['tasks'] == [row['tasks'][0]]
    assert kept['gold'] == {'head': {}}
    assert kept['text'] == row['text'] and kept['split'] == 'dev'
    assert kept['artifact'] == 'single_object' and in_scope(kept)
    assert audit[0]['reason'] == 'public_layout_position_task'


@pytest.mark.parametrize('text', ['Name\tQty\tPrice\nWidget\t2\t10',
                                  '|Name|Qty|\n|Widget|2|',
                                  'Name  Qty  Price\nWidget  2  10'])
def test_table_records_excluded(text):
    assert not in_scope({'family': 'invoices', 'source': 'tractor_authored_documents',
                         'text': text, 'tasks': [{'kind': 'records'}]})


def test_explicit_prose_records_stay_in_scope():
    assert in_scope({'family': 'careers', 'source': 'tractor_authored_documents',
                     'text': 'Anna joined Example GmbH on 1 March. Ben joined Other GmbH in May.',
                     'tasks': [{'kind': 'records'}]})


def test_incidental_table_is_not_a_header_extraction_task():
    assert in_scope({'family': 'invoices', 'source': 'cord',
                     'text': 'Name\tQty\tPrice\nWidget\t2\t10\nTotal: 20',
                     'tasks': [{'kind': 'entities'}]})
