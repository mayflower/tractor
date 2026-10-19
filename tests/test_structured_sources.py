import pytest

from tractor.jsonio import dumps
from tractor.structured_sources import adapt, evidence, group_near_documents, parse_resume


def text_row(text, schema, output):
    return {'text': '### Input: ' + text + '\n### Schema: ' + dumps(schema) + '\n### Output: ' + dumps(output)}


def test_projection_omits_hallucinated_null_and_derived_fields():
    row = text_row('Anna works at Acme.', {'name': 'string', 'company': 'string', 'missing': 'string|null',
                   'summary': 'string'}, {'name': 'Anna', 'company': 'Invented Inc', 'missing': None,
                                         'summary': 'Anna works at Acme.'})
    e, counts = adapt(row, 'text_json', 1)
    assert e.text_origin == 'unknown' and e.label_origin == 'unknown'
    assert e.annotation == 'silver'
    assert list(e.tasks[0].fields.values()) == ['name']
    assert e.gold['document'].records[0]['f0'].items[0].text == 'Anna'
    assert counts['unverified_null_or_empty_omitted'] == 1


def test_partial_list_is_not_silently_relabelled_as_complete():
    e, _ = adapt(text_row('Python and SQL.', {'skills': ['string']},
                         {'skills': ['Python', 'Imaginary']}), 'text_json', 1)
    assert e is None


def test_nested_records_preserve_pairing_and_drop_bad_field_for_all_rows():
    row = text_row('Alpha costs 12. Beta costs 34.',
                   {'items': [{'name': 'string', 'price': 'number', 'category': 'string'}]},
                   {'items': [{'name': 'Alpha', 'price': 12, 'category': 'X'},
                              {'name': 'Beta', 'price': 34, 'category': 'Y'}]})
    e, _ = adapt(row, 'text_json', 1)
    assert e.tasks[0].kind == 'records'
    records = e.gold[e.tasks[0].id].records
    assert [[a.items[0].text for a in r.values()] for r in records] == [['Alpha', '12'], ['Beta', '34']]


@pytest.mark.parametrize('text,value', [('Anna met Anna', 'Anna'), ('foobar', 'foo'),
                                      ('20.5', 20), ('1,020', 20), ('20e3', 20), ('True', True)])
def test_unreliable_evidence_rejected(text, value):
    with pytest.raises(ValueError):
        evidence(text, value)


def test_resume_uses_only_user_text_not_assistant_output():
    row = {'text': '<|im_start|>system\nInstructions<|im_end|>\n'
           '<|im_start|>user\nResume:\nEngineer at Acme.<|im_end|>\n'
           '<|im_start|>assistant\n<think>\n</think>\n'
           '{"current_title":"Engineer","current_company":"Invented","seniority":"senior"}'
           '<|im_end|>'}
    text, _, _ = parse_resume(row)
    assert text == 'Engineer at Acme.'
    e, _ = adapt(row, 'resume_json', 1)
    assert e.text_origin == 'real' and e.label_origin == 'teacher'
    assert len(e.tasks[0].fields) == 1


def test_same_source_text_cannot_cross_splits_between_supplements():
    e, _ = adapt(text_row('Engineer at Acme.', {'current_title': 'string'}, {'current_title': 'Engineer'}),
                 'text_json', 1)
    cv = {'text': '<|im_start|>system\nInstructions<|im_end|>'
          '<|im_start|>user\nResume:\nEngineer at Acme.<|im_end|>'
          '<|im_start|>assistant\n{"current_title":"Engineer"}<|im_end|>'}
    other, _ = adapt(cv, 'resume_json', 2)
    assert e.source_group == other.source_group and e.split == other.split


def test_near_duplicate_documents_share_one_split():
    text = 'Anna visited the quiet old library near the central railway station on Monday.'
    first, _ = adapt(text_row(text, {'name': 'string'}, {'name': 'Anna'}), 'text_json', 1)
    second, _ = adapt(text_row(text + ' Afterwards she left.', {'name': 'string'}, {'name': 'Anna'}), 'text_json', 2)
    a, b = group_near_documents([first, second])
    assert a.source_group == b.source_group and a.split == b.split
