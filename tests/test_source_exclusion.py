import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tractor.document_text_scope import restrict
from tractor.extraction_scope import in_scope
from tractor.pipeline_evaluation import typed_requests_for


@pytest.mark.parametrize('split', ['train', 'dev', 'calibration', 'test'])
@pytest.mark.parametrize('source', ['cord', 'CORD', 'cord-v2', 'naver-clova-ix/cord-v2'])
def test_cord_headers_and_records_cannot_reenter_any_split(split, source):
    row = {'id': 'receipt', 'source': source, 'family': 'invoices', 'split': split,
           'tasks': [{'id': 'header', 'kind': 'entities'}, {'id': 'items', 'kind': 'records'}]}
    assert not in_scope(row)
    kept, removed = restrict(row)
    assert kept is None and len(removed) == 2
    assert {r['reason'] for r in removed} == {'excluded_source'}
    with pytest.raises(ValueError, match='CORD'):
        typed_requests_for(SimpleNamespace(**row))


def test_cord_cannot_be_restored_by_renaming_source():
    row = {'id': 'cord:train:receipt:window:0:100', 'source': 'import', 'family': 'invoices'}
    assert not in_scope(row)


def test_registry_and_importers_no_longer_offer_cord():
    from tractor import document_sources

    registry = json.loads(Path('configs/document-sources.json').read_text())
    assert not any('cord' in r['source'].lower() for r in registry)
    assert not hasattr(document_sources, 'cord')


def test_scoring_rejects_cord_instead_of_turning_it_into_model_errors():
    from tractor.pipeline_evaluation import evaluate_pipeline

    with pytest.raises(ValueError, match='clean the dataset'):
        evaluate_pipeline([SimpleNamespace(source='cord')], {}, {})


def test_evaluator_rejects_cord_before_loading_a_model(monkeypatch, tmp_path):
    from tractor import document_eval

    monkeypatch.setattr(document_eval, 'read_episodes', lambda _: [SimpleNamespace(source='cord')])
    with pytest.raises(ValueError, match='clean the dataset'):
        document_eval.run('no-model', 'no-data', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
