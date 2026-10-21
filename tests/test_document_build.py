from pathlib import Path

import pytest

from tractor.document_build import apply_facts, approved, plan_item, reconstruct
from tractor.document_sources import kleister, vrdu, windows
from tractor.jsonio import loads


def scenario(shape='single_object'):
    spec = loads(Path('schemas/extraction/invoices.json').read_text())
    return plan_item(spec, 'train', 1, 42, shape)


def rendered(item):
    return {'id': item['id'], 'marked_text': 'This is a fictional document with enough contextual wording for the test to check the original character offsets. '
            + ' '.join(f'<{f["marker"]}>{f["value"]}</{f["marker"]}>' for f in item['facts'])
            + ' Please retain this document with your original records for future reference.'}


def test_marker_reconstruction_preserves_offsets_and_never_emits_markers():
    item = scenario()
    e = reconstruct(item, rendered(item))
    assert '<v' not in e.text
    for row in e.gold['document'].records:
        for a in row.values():
            for s in a.items:
                assert e.text[s.start:s.end] == s.text


def test_changed_fact_or_marker_is_rejected():
    item = scenario()
    result = rendered(item)
    result['marked_text'] = result['marked_text'].replace('<v0>', '<v999>')
    with pytest.raises(ValueError):
        reconstruct(item, result)


def test_missing_marker_recovery_preserves_text_and_refuses_ambiguous_literals():
    from tractor.document_build import restore_unique_markers

    item = {'facts': [{'marker': 'v0', 'value': 'Cedar Ltd'}]}
    raw = {'id': 'x', 'marked_text': 'Supplier: Cedar Ltd.'}
    fixed, markers = restore_unique_markers(item, raw)
    assert fixed['marked_text'] == 'Supplier: <v0>Cedar Ltd</v0>.' and markers == ['v0']
    ambiguous = dict(raw, marked_text='Cedar Ltd supplies Cedar Ltd.')
    assert restore_unique_markers(item, ambiguous) == (ambiguous, [])


def test_fact_proposal_cannot_change_schema_split_or_identity():
    item = scenario()
    proposed = {'id': item['id'], 'facts': [{'marker': f['marker'], 'value': f['value']} for f in item['facts']]}
    result = apply_facts(item, proposed)
    assert result['tasks'] == item['tasks'] and result['split'] == item['split']
    proposed['facts'].pop()
    with pytest.raises(ValueError):
        apply_facts(item, proposed)


def test_unqueried_context_facts_do_not_expand_the_gold_schema():
    item = scenario()
    assert any(f['path'].startswith('context.') for f in item['facts'])
    e = reconstruct(item, rendered(item))
    assert set(e.gold['document'].records[0]) == set(item['tasks'][0]['fields'])


def test_unqualified_dataset_is_refused_before_model_loading(tmp_path):
    from tractor.train import train

    (tmp_path / 'manifest.json').write_text('{"training_ready": false}')
    with pytest.raises(ValueError, match='not training-ready'):
        train('nonexistent', tmp_path, 'nonexistent')


def test_bulk_authoring_cannot_bypass_realism_audit(tmp_path):
    from tractor.document_build import check_scale_gate

    check_scale_gate(tmp_path, 64)
    with pytest.raises(ValueError, match='requires a passing realism'):
        check_scale_gate(tmp_path, 30000)


def test_review_requires_all_explicit_boolean_checks_and_no_issues():
    flags = dict.fromkeys(['supported', 'correct_roles', 'complete', 'natural_document', 'no_reasoning_required'], True)
    assert approved(flags | {'issues': []})
    assert not approved(flags | {'natural_document': 'true', 'issues': []})
    assert not approved(flags | {'issues': ['implausible repeated values']})


def test_kleister_never_trains_normalized_dates_as_literal_values():
    text = 'Signed by Cedar Ltd on April 18, 2001.'
    e, excluded = kleister('doc.pdf\tparty effective_date\t'+text, 'party=Cedar_Ltd effective_date=2001-04-18', 'kleister-nda', 'train')
    assert excluded == {'effective_date': 'normalized_or_unmatched_gold'}
    assert set(e.tasks[0].fields) == {'party'}
    assert e.gold['document'].records[0]['party'].items[0].text == 'Cedar Ltd'


def test_vrdu_must_match_original_offsets():
    row = {'filename': 'a.pdf', 'ocr': {'text': 'Number 123'}, 'annotations': [['number', [['123', [], [[7, 10]]]]]]}
    e = vrdu(row, 'dev', 'forms')
    assert e.gold['header'].records[0]['number'].items[0].start == 7
    row['annotations'][0][1][0][0] = '456'
    with pytest.raises(ValueError):
        vrdu(row, 'dev', 'forms')


def test_windows_do_not_use_gold_for_boundaries_and_keep_document_groups():
    e = vrdu({'filename': 'a.pdf', 'ocr': {'text': 'x '*40+'123'},
              'annotations': [['number', [['123', [], [[80, 83]]]]]]}, 'dev', 'forms')
    chunks = windows(e, max_chars=50)
    assert ''.join(c.text for c in chunks) == e.text
    assert all(c.source_group == e.source_group and c.split == 'dev' for c in chunks)
    assert chunks[-1].gold['header'].records[0]['number'].items[0].text == '123'


def test_all_domain_schemas_and_quotas_are_present():
    config = loads(Path('configs/document-build.json').read_text())
    assert len(config['domains']) == 16 and sum(config['domains'].values()) == 100
    for name in config['domains']:
        s = loads(Path('schemas/extraction', name+'.json').read_text())
        assert s['anchor'] in s['record']
        assert s['schema']['properties']['records']['items']['type'] == 'object'


def test_all_classification_and_relations_remain_out_of_scope():
    from tractor.extraction_scope import in_scope

    assert not in_scope({'family': 'classification_approval', 'source': 'x'})
    assert not in_scope({'family': 'invoices', 'source': 'x', 'tasks': [{'kind': 'relations'}]})
    assert not in_scope({'family': 'invoices', 'source': 'x', 'tasks': [{'kind': 'classification'}]})


def test_repeated_unambiguous_entity_mentions_have_complete_offsets():
    item = scenario()
    fact = next(f for f in item['facts'] if f['path'].startswith('document.') and f['path'].split('.')[-1] in {'supplier', 'customer', 'number'})
    result = rendered(item)
    result['marked_text'] += ' Repeated reference: ' + fact['value'].upper()
    e = reconstruct(item, result)
    key = fact['path'].split('.')[-1]
    assert len(e.gold['document'].records[0][key].items) == 2
    assert all(e.text[s.start:s.end] == s.text for s in e.gold['document'].records[0][key].items)
