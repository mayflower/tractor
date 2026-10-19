from copy import deepcopy

import pytest

from tractor.corpus_annotations import annotate
from tractor.document_sources import envelope
from tractor.jsonio import dumps, loads
from tractor.pipeline_evaluation import typed_requests_for
from tractor.typed_corpus import build, evidence, migrate
from tractor.typed_corpus_policy import choose, currency_context


def episode(value, field='date', family='operations', source='tractor_authored_documents', language='de'):
    return envelope(source, 'migration:test', 'train', family, language, value,
        [{'id': 'doc', 'kind': 'entities', 'description': 'Extraction', 'fields': {field: field}}],
        {'doc': {'complete': True, 'records': [{field: {'status': 'present',
            'items': [{'start': 0, 'end': len(value), 'text': value}]}}]}})


@pytest.mark.parametrize(('text', 'rule', 'expected'), [
    ('03/2019', {'kind': 'date', 'allow_partial': True}, ['2019-03']),
    ('14.10.2026, 07:45 Uhr', {'kind': 'datetime', 'date_order': 'DMY'}, ['2026-10-14T07:45']),
    ('n = 186', {'kind': 'number'}, ['186']),
    ('n = 60 specimens (12 formulations × 5 replicates)', {'kind': 'number'}, ['60']),
    ('240 Millionen Euro', {'kind': 'money', 'currency': 'EUR'}, ['240000000']),
    ('1,850.50 EUR', {'kind': 'money', 'currency': 'EUR'}, ['1850.5']),
    ('DE89 3704 0044 0532 0130 00', {'kind': 'iban'}, ['DE89370400440532013000']),
    ('HTTPS://EXAMPLE.ORG/Shop?A=B', {'kind': 'url'}, ['https://example.org/Shop?A=B']),
    ('Dr.\nHelena Vogt', {'kind': 'person'}, ['Dr. Helena Vogt']),
    ('2200 g', {'kind': 'quantity', 'unit': 'kg'}, ['2.2']),
    ('01.09.2026 – 30.09.2026', {'kind': 'interval', 'date_order': 'DMY'}, ['2026-09-01/2026-09-30']),
])
def test_independent_annotation_oracles(text, rule, expected):
    assert annotate(text, rule) == expected


@pytest.mark.parametrize(('text', 'rule'), [
    ('03/04/2026', {'kind': 'date'}),
    ('1,234', {'kind': 'number'}),
    ('DE45 5001 0517 5407 3249 31', {'kind': 'iban'}),
    ('fiscal year ended 31 March 2025', {'kind': 'interval'}),
    ('12 × 5', {'kind': 'number'}),
    ('n = 60 to 70', {'kind': 'number'}),
    ('n = 60 or 70 participants', {'kind': 'number'}),
    ('020 7946 0018', {'kind': 'phone'}),
])
def test_unknown_or_invalid_target_is_not_manufactured(text, rule):
    with pytest.raises(ValueError):
        annotate(text, rule)


def test_types_do_not_depend_on_answer_presence_or_text_of_gold():
    present = episode('n = 186', 'sample_size', 'research')
    absent = deepcopy(present)
    answer = absent.gold['doc'].records[0]['sample_size']
    answer.status, answer.items = 'absent', []
    a, _ = migrate(present)
    b, _ = migrate(absent)
    assert typed_requests_for(a) == typed_requests_for(b)
    assert a.gold['doc'].records[0]['sample_size'].final_values == ['186']


def test_datetime_repairs_missing_annotation_and_keeps_evidence_and_idempotence():
    original = episode('14.10.2026, 07:45 Uhr')
    from tractor.field_contracts import Processing
    original.processing = {'doc': {'date': Processing(kind='date', date_order='DMY', allow_partial=True)}}
    original.gold['doc'].records[0]['date'].final_annotation_missing = True
    result, events = migrate(original)
    assert result.processing['doc']['date'].kind == 'datetime'
    assert result.gold['doc'].records[0]['date'].final_values == ['2026-10-14T07:45']
    assert evidence(result.model_dump()) == evidence(original.model_dump())
    assert any(e['event'] == 'annotation_recovered' for e in events)
    twice, events = migrate(result)
    assert twice == result and not events


def test_swde_names_use_source_schema_not_shared_key():
    for vertical, expected in [('restaurant', 'organization'), ('university', 'organization'), ('nbaplayer', 'person')]:
        original = episode('Example Name', 'name', 'contacts', source='swde')
        original.source_group = f'swde:{vertical}:site:1'
        result, _ = migrate(original)
        assert result.processing['doc']['name'].kind == expected


def test_mixed_name_and_unknown_currency_are_reported_without_guessing():
    result, events = migrate(episode('Example Name', 'name', 'careers'))
    assert result.processing['doc']['name'].kind == 'literal'
    assert any(e['event'] == 'contract_gap' for e in events)
    assert currency_context('$100') is None
    assert currency_context('100 EUR and 20 GBP') is None
    assert currency_context('100 Euro') == 'EUR'
    assert currency_context('100 Euro and $20') is None


def test_notice_clause_is_not_reduced_to_a_duration():
    result, events = migrate(episode('drei Monate zum Quartalsende', 'notice', 'contracts'))
    assert result.processing['doc']['notice'].kind == 'literal'
    assert result.gold['doc'].records[0]['notice'].final_values == ['drei Monate zum Quartalsende']
    assert any(e.get('reason') == 'notice_clause_includes_calendar_conditions_not_just_duration' for e in events)


def test_missing_annotation_stays_unscored_not_an_empty_positive():
    result, events = migrate(episode('DE45 5001 0517 5407 3249 31', 'iban', 'invoices'))
    answer = result.gold['doc'].records[0]['iban']
    assert answer.status == 'present' and answer.items
    assert answer.final_annotation_missing and answer.final_values is None
    assert any(e['event'] == 'annotation_gap' for e in events)


def test_whole_corpus_migration_preserves_splits_and_native_bytes(tmp_path):
    from tractor.coverage_assembly import sha
    source, output = tmp_path / 'source', tmp_path / 'typed'
    source.mkdir()
    for split in ('train', 'dev', 'calibration', 'test'):
        row = episode('Dr. Helena Vogt', 'author', 'research')
        row.split = split
        row.annotation = 'silver'
        row.text_origin = 'synthetic' if split in {'train', 'dev'} else 'real'
        row.label_origin = 'teacher' if split in {'train', 'dev'} else 'deterministic'
        (source / f'{split}.episodes.jsonl').write_text(dumps(row.model_dump())+'\n')
        (source / f'{split}.native.jsonl').write_text('{"frozen": "native supervision"}\n')
    files = {p.name: sha(p) for p in source.iterdir()}
    (source / 'manifest.json').write_text(dumps({'training_ready': True, 'files': files}))
    report = build(source, output)
    assert len(report['splits']) == 4
    for split in ('train', 'dev', 'calibration', 'test'):
        row = loads((output / f'{split}.episodes.jsonl').read_text())
        assert row['split'] == split
        assert row['processing']['doc']['author']['kind'] == 'person'
        from tractor.episodes import Episode
        restored = Episode.model_validate(row)
        assert restored.text_origin == ('synthetic' if split in {'train', 'dev'} else 'real')
        assert restored.label_origin == ('teacher' if split in {'train', 'dev'} else 'deterministic')
        assert (output / f'{split}.native.jsonl').read_bytes() == (source / f'{split}.native.jsonl').read_bytes()
        assert sha(source / f'{split}.episodes.jsonl') == files[f'{split}.episodes.jsonl']


def test_policy_has_no_gold_parameter_and_keeps_declared_country():
    schema = dict(source='swde', source_group='swde:restaurant:x:1', family='contacts',
                  text='Phone: +44 20 7946 0018', language='en', locale=None)
    existing = dict(kind='phone', country='GB', cardinality='optional_one')
    rule, debt = choose(schema, {'fields': {'phone': 'Phone'}, 'anchor': None}, 'phone', existing)
    assert rule == existing and debt is None
