import pytest

from tractor.document_duplicates import Index
from tractor.document_quality import evaluate
from tractor.document_repair import safe_windows
from tractor.document_sources import envelope, vrdu
from tractor.document_training import selection_score


def test_public_masking_cannot_delete_distinct_receipts_but_literal_duplicate_can():
    def row(ident, text, split):
        return {'id': ident, 'source_group': ident, 'split': split, 'text_origin': 'real', 'text': text,
                'gold': {'x': {'records': [{'v': {'items': [{'start': 0, 'end': len(text)}]}}]}}}
    index = Index()
    assert index.check(row('held', 'Coffee 12.00', 'dev'), cross_only=True) == []
    assert index.check(row('train', 'Tea 8.50', 'train'), cross_only=True) == []
    assert index.check(row('copy', 'Coffee 12.00', 'train'), cross_only=True)


def test_record_gate_does_not_count_correct_header_as_correct_record():
    def value(start, text):
        return {'status': 'present', 'items': [{'start': start, 'end': start+len(text), 'text': text}]}
    e = envelope('x', 'a', 'dev', 'invoices', 'en', 'Invoice A\nTea',
                 [{'id': 'header', 'kind': 'entities', 'description': 'Header', 'fields': {'number': 'Number'}},
                  {'id': 'items', 'kind': 'records', 'description': 'Items', 'fields': {'name': 'Item'}, 'anchor': 'name'}],
                 {'header': {'complete': True, 'records': [{'number': value(8, 'A')}]},
                  'items': {'complete': True, 'records': [{'name': value(10, 'Tea')}]}})
    policy = {'domains': ['invoices'], 'shapes': ['repeated_records'], 'required_origins': ['real/human'],
              'minimum_exclusive': .7, 'qualification': 'test'}
    result = evaluate([e], {e.id: {'prediction': {'entities': {'number': [{'start': 8, 'end': 9, 'text': 'A'}]}}}}, policy)
    assert result['checks']['shape:repeated_records:real/human:record_f1']['score'] == 0
    assert result['scores']['task:records:real/human']['counts']['gold_records'] == 1


def test_zero_floor_no_longer_freezes_checkpoint_selection():
    def report(value):
        return {'passed': False, 'checks': {'record': {'score': 0}, 'fields': {'score': value}, 'missing': {'score': None}}}
    assert selection_score(report(.8)) > selection_score(report(.4)) > 0
    assert not report(.8)['passed']


def test_vrdu_common_columns_do_not_create_competing_partial_tables():
    def cell(text, start):
        return [text, [], [[start, start+len(text)]]]
    row = {'filename': 'x', 'ocr': {'text': 'Tea 10\nCake'}, 'annotations': [
        [['program_desc', 'sub_amount'], [[cell('Tea', 0), cell('10', 4)]]],
        [['program_desc'], [[cell('Cake', 7)]]]]}
    e = vrdu(row, 'train', 'invoices')
    assert len(e.tasks) == 1
    assert set(e.tasks[0].fields) == {'program_desc'}
    assert len(e.gold['items'].records) == 2


def test_partial_date_annotation_is_rejected():
    row = {'filename': 'x', 'ocr': {'text': '05/20/20'},
           'annotations': [['file_date', [['05/20/', [], [[0, 6]]]]]]}
    with pytest.raises(ValueError, match='cuts through'):
        vrdu(row, 'dev', 'forms')


def test_safe_window_artifact_tracks_actual_tasks():
    e = vrdu({'filename': 'x', 'ocr': {'text': 'Number 123'},
              'annotations': [['number', [['123', [], [[7, 10]]]]]]}, 'train', 'forms')
    result = list(safe_windows(e))
    assert len(result) == 1 and result[0].artifact == 'single_object'
    assert result[0].source_group == e.source_group


def test_public_representative_contains_records_not_just_header():
    from tractor.document_mixture import key, real_representatives

    header = {'source': 'vrdu', 'source_group': 'doc', 'id': 'header', 'family': 'invoices',
              'artifact': 'repeated_records', 'tasks': [{'kind': 'entities'}]}
    records = dict(header, id='positions', tasks=[{'kind': 'entities'}, {'kind': 'records'}])
    assert key(header) == ('invoices', 'single_object')
    chosen = real_representatives([header, records], 42)
    assert list(chosen) == [('invoices', 'repeated_records')]
    assert chosen['invoices', 'repeated_records']['doc'][1] == 'positions'


def test_scalar_schema_is_explicit_and_renaming_preserves_it():
    from tractor.episodes import Task, compile_schema, schema_variant

    task = Task(id='items', kind='records', description='Rows', fields={'name': 'Item', 'price': 'Price'},
                anchor='name', cardinalities={'name': 'required_one', 'price': 'optional_one'})
    built = compile_schema([task]).build()
    assert built['record_metadata']['items']['fields']['price']['cardinality'] == 'optional_one'
    e = vrdu({'filename': 'x', 'ocr': {'text': 'Tea 10'}, 'annotations': [
        [['program_desc', 'sub_amount'], [[['Tea', [], [[0, 3]]], ['10', [], [[4, 6]]]]]]]}, 'train', 'invoices')
    e.tasks[0].cardinalities = {'program_desc': 'required_one', 'sub_amount': 'optional_one'}
    changed = schema_variant(e, 7)
    assert set(changed.tasks[0].cardinalities) == set(changed.tasks[0].fields)


def test_larger_split_cannot_reassign_prior_heldout_or_training_groups(tmp_path):
    import hashlib
    import json

    from tractor.source_contracts import split_lookup

    files = {}
    for name, content in [('large', {'train': ['held', 'new'], 'valid': ['trained']}),
                          ('old', {'train': ['trained'], 'test': ['held']})]:
        rel = f'ad-buy-form/few_shot-splits/{name}.json'
        p = tmp_path/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(content))
        files[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    spec = {'path': str(tmp_path), 'files': files,
            'active_splits': {'ad-buy-form': 'ad-buy-form/few_shot-splits/large.json'}}
    result = split_lookup(spec, 'ad-buy-form')
    assert result == {'held': 'test', 'new': 'train', 'trained': 'train'}


def test_document_plan_is_no_longer_limited_to_two_or_three_records():
    import json
    from pathlib import Path

    from tractor.document_build import plan_item

    spec = json.loads(Path('schemas/extraction/clinical.json').read_text())
    counts = {len(plan_item(spec, 'train', i, 1, 'repeated_records')['rows']['records']) for i in range(6)}
    assert counts == {1, 2, 3, 4, 6, 8}
    assert plan_item(spec, 'train', 0, 1, 'repeated_records', record_count=0)['rows']['records'] == []
