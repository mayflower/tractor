from tractor.swde_sources import adapt, visible_text


def test_visible_text_drops_scripts_and_never_uses_labels_for_selection():
    raw = '<html><head><title>Hidden title</title></head><body><script>secret</script><p>Acme &amp; Sons</p><p>London</p></body></html>'
    assert visible_text(raw) == 'Acme & Sons\nLondon'
    a, removed = adapt(raw, {'company': ['Acme & Sons'], 'location': ['London'], 'date_posted': ['2026-10-01']},
                       'job', 'example', '0001', 'train')
    assert a.text_origin == 'real' and a.label_origin == 'deterministic'
    assert removed == {'date_posted': 'not_literal_in_visible_text'}
    assert 'date_posted' not in a.tasks[0].fields
    for row in a.gold['document'].records:
        for answer in row.values():
            assert all(a.text[s.start:s.end] == s.text for s in answer.items)


def test_identifier_substrings_are_not_annotated_as_independent_values():
    raw = '<p>ISBN 9780805087444</p><p>Inventory KB-9780805087444</p><p>Title: A Book</p>'
    e, _ = adapt(raw, {'isbn_13': ['9780805087444'], 'title': ['A Book']}, 'book', 's', '1', 'train')
    assert len(e.gold['document'].records[0]['isbn_13'].items) == 1
