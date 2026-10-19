from copy import deepcopy

import pytest

from tractor.corpus_provenance import restore
from tractor.document_sources import envelope
from tractor.episodes import Episode
from tractor.pipeline_evaluation import typed_requests_for


def reference():
    e = envelope('tractor_authored_documents', 'provenance:1', 'dev', 'contacts', 'de', 'Ada Lovelace',
        [{'id': 'doc', 'kind': 'entities', 'description': 'Contact', 'fields': {'name': 'Person name'}}],
        {'doc': {'complete': True, 'records': [{'name': {'status': 'present',
            'items': [{'start': 0, 'end': 12, 'text': 'Ada Lovelace'}]}}]}})
    e.annotation, e.text_origin, e.label_origin = 'silver', 'synthetic', 'teacher'
    return e.model_dump()


def test_restore_missing_origin_preserves_typed_request_and_supervision():
    original = reference()
    compact = deepcopy(original)
    del compact['text_origin']
    assert Episode.model_validate(compact).text_origin == 'real'
    restored = Episode.model_validate(restore(compact, original))
    assert restored == Episode.model_validate(original)
    assert typed_requests_for(restored) == typed_requests_for(Episode.model_validate(compact))
    assert restored.gold == Episode.model_validate(compact).gold


@pytest.mark.parametrize('key', ['id', 'source', 'source_group', 'split', 'text'])
def test_restore_rejects_reference_mismatch(key):
    original = reference()
    wrong = {**original, key: 'different'}
    with pytest.raises(ValueError, match='identity/text/split'):
        restore(original, wrong)
