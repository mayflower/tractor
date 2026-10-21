import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from tractor.document_sources import envelope
from tractor.service import extract
from tractor.types import ExtractRequest


@pytest.fixture
def diagnostic():
    path = Path(__file__).parents[1] / 'scripts' / 'compare-biqwen-schemas.py'
    spec = importlib.util.spec_from_file_location('biqwen_schema_diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair():
    episode = envelope('tractor_authored_documents', 'schema-test', 'dev', 'metadata', 'en', 'Ada',
                       [{'id': 'document', 'kind': 'entities', 'description': 'Person',
                         'fields': {'name': 'Person'}}],
                       {'document': {'complete': True, 'records': [{'name': {'status': 'present',
                         'items': [{'start': 0, 'end': 3, 'text': 'Ada'}]}}]}})
    native = {'text': 'Ada', 'metadata': {'split': 'validation', 'source': episode.source},
              'schema': {'fields': [{'id': 't0.name', 'description': 'Person'}],
                         'tables': [], 'choices': []}}
    return episode, native


@pytest.mark.parametrize('split', ['train', 'calibration', 'test_human', 'test_id'])
def test_schema_diagnostic_rejects_non_validation_before_inference(diagnostic, split):
    episode, native = pair()
    native['metadata']['split'] = split
    with pytest.raises(ValueError, match='Only native validation'):
        diagnostic.validate_contract(episode, native)


def test_schema_diagnostic_rejects_field_mapping_and_indirection(diagnostic):
    episode, native = pair()
    diagnostic.validate_contract(episode, native)
    native['schema']['fields'][0]['id'] = 't1.name'
    with pytest.raises(ValueError, match='field mapping'):
        diagnostic.validate_contract(episode, native)
    native['schema']['fields'][0]['id'] = 't0.name'
    native['schema']['fields'][0]['resolve_references'] = True
    with pytest.raises(ValueError, match='reference resolution'):
        diagnostic.validate_contract(episode, native)


def test_native_schema_outputs_use_public_normalization_and_preserve_ambiguity(diagnostic):
    request = ExtractRequest(text='Alpha: 10 cm, 12 cm.', many=True, anchor='name', fields={
        'name': {'type': 'text', 'description': 'Product'},
        'width': {'type': 'quantity', 'description': 'Width', 'unit': 'mm'},
    })
    episode = SimpleNamespace(tasks=[SimpleNamespace(fields=request.fields)])
    raw = {'tables': {'t0': [{'anchor': [0, 5], 'fields': {'t0.width': [[7, 12], [14, 19]]}}]}}
    result = extract(diagnostic.NativeEvidence(episode, raw, False), request)
    assert result['data'] == [{'name': 'Alpha', 'width': [100, 120]}]
    assert result['issues'] == [{'field': 'width', 'index': 0, 'code': 'ambiguous_value'}]
