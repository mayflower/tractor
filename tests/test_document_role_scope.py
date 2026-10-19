from tractor.document_role_scope import explicit_local_role, restrict


def row(source, text, key, value, family='invoices'):
    answer = {'status': 'present', 'items': [{'start': text.index(value), 'end': text.index(value)+len(value), 'text': value}]} if value else {'status': 'absent', 'items': []}
    return {'id': 'example', 'source': source, 'split': 'train', 'family': family, 'text': text,
            'tasks': [{'id': 'header', 'kind': 'entities', 'fields': {key: key}}],
            'gold': {'header': {'complete': True, 'records': [{key: answer}]}}}


def test_missing_source_annotation_never_becomes_negative():
    value = row('swde', '3.0L engine', 'engine', None)
    kept, audit = restrict(value)
    assert kept is None and audit[0]['reason'] == 'source_absence_not_verified_locally'
    assert value['gold']['header']['records'][0]['engine']['status'] == 'absent'


def test_jurisdiction_needs_role_at_exact_occurrence():
    text = 'A Delaware corporation. Governed by the laws of Delaware.'
    assert not explicit_local_role(text, 'jurisdiction', {'start': 2}, 'kleister-nda')
    assert explicit_local_role(text, 'jurisdiction', {'start': text.rindex('Delaware')}, 'kleister-nda')


def test_header_columns_are_not_local_roles():
    value = row('vrdu', 'Gross Amount\nCommission\n$50\n$5', 'gross_amount', '$50')
    assert restrict(value)[0] is None
    value = row('vrdu', 'Gross Amount: $50', 'gross_amount', '$50')
    assert restrict(value)[0] == value


def test_financial_metadata_does_not_claim_metric_coverage():
    value = row('kleister-charity', 'Example Foundation', 'charity_name', 'Example Foundation', 'financial_reports')
    kept, audit = restrict(value)
    assert kept['family'] == 'metadata' and audit
    assert restrict(kept)[1] == []
    value = row('kleister-charity', 'Annual income: 123', 'income_annually_in_british_pounds', '123', 'financial_reports')
    assert restrict(value)[0]['family'] == 'financial_reports'


def test_authored_absences_and_prose_records_unchanged():
    value = row('tractor_authored_documents', 'No phone supplied.', 'phone', None)
    assert restrict(value) == (value, [])


def test_previous_public_qualification_cannot_bypass_role_audit(tmp_path):
    import json

    import pytest

    from tractor.document_mixture import verified
    (tmp_path/'manifest.json').write_text(json.dumps({'training_ready': True, 'unique_real_train': 206334}))
    with pytest.raises(ValueError, match='local-role audit'):
        verified(tmp_path)


def test_removed_field_is_not_a_native_negative():
    from tractor.document_sources import envelope
    from tractor.episodes import Episode, native_episode

    e = envelope('swde', 'id', 'train', 'products', 'en', 'Car X engine 3.0L',
                 [{'id':'document','kind':'entities','description':'Explicit car fields',
                   'fields':{'model':'Car model','engine':'Engine'}}],
                 {'document':{'complete':True,'records':[{
                     'model':{'status':'present','items':[{'start':0,'end':5,'text':'Car X'}]},
                     'engine':{'status':'absent','items':[]}}]}})
    kept, _ = restrict(e.model_dump())
    native, skipped = native_episode(Episode.model_validate(kept))
    assert not skipped
    assert 'engine' not in native['output']['entities']
    assert native['output']['entities']['model'][0]['text'] == 'Car X'


def test_financial_repeated_numbers_need_explicit_total_role():
    text = 'Membership fees 110.00\nTotal annual income: £110.00'
    assert not explicit_local_role(text, 'income_annually_in_british_pounds', {'start': text.index('110')}, 'kleister-charity')
    assert explicit_local_role(text, 'income_annually_in_british_pounds', {'start': text.rindex('110')}, 'kleister-charity')


def test_generic_name_label_is_not_sufficient_registrant_role():
    value = row('vrdu', 'Name\nOther company', 'registrant_name', 'Other company')
    assert restrict(value)[0] is None
