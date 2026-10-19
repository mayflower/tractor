from copy import deepcopy

from tractor.document_pool import projected


def row(source='few_nerd', status='absent'):
    return {'source': source, 'text_origin': 'real', 'family': 'named_entities',
            'tasks': [{'id': 'entities', 'kind': 'entities', 'fields': {'person': 'Named person'}}],
            'gold': {'entities': {'complete': True, 'records': [{'person': {'status': status, 'items': []}}]}}}


def test_source_negatives_are_not_arbitrary_empty_document_windows():
    source = row()
    assert projected(deepcopy(source), True) == source
    assert projected(row('swde')) is None
    assert projected(row(status='unknown'), True) is None
    incomplete = row()
    incomplete['gold']['entities']['complete'] = False
    assert projected(incomplete, True) is None


def test_historical_synthesis_and_excluded_sources_cannot_reenter_pool():
    assert projected(row('massive')) is None
    assert projected(row('teacher'), True) is None
    synthetic = row()
    synthetic['text_origin'] = 'synthetic'
    assert projected(synthetic, True) is None
