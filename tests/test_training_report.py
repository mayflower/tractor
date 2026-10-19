from tractor.jsonio import dumps
from tractor.training_report import pilot_report


def test_full_request_regression_excludes_otherwise_better_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr('tractor.training_report.clustered_difference', lambda *args: {})
    for arm in ('B0', 'B1', 'P0', 'P1', 'P2', 'P3'):
        native = {'macro_field_f1': .8 if arm == 'P3' else .4,
                  'overall': {'request_exact': .5}}
        product = {'field_f1': .99, 'exact_record_rate': .98,
                   'precision': .98,
                   'exact_request_rate': .9 if arm == 'P3' else .98}
        for kind, data in [('native', native), ('product', [{'metrics': product}])]:
            path = tmp_path / arm / kind
            path.mkdir(parents=True)
            (path / 'summary.json').write_text(dumps(data))
    assert pilot_report(tmp_path)['selected'] == 'P3'
    (tmp_path / 'design.json').write_text(dumps({
        'product_regression_metrics': ['field_f1', 'exact_record_rate', 'exact_request_rate']}))
    report = pilot_report(tmp_path)
    assert report['selected'] == 'B1'
    assert not report['eligible']
    (tmp_path / 'design.json').write_text(dumps({'minimum_product_precision': .99}))
    assert pilot_report(tmp_path)['selected'] == 'B1'
    # An explicit absolute floor replaces relative regression guards when disabled.
    (tmp_path / 'design.json').write_text(dumps({
        'product_regression_metrics': [],
        'minimum_product_metrics': {'exact_request_rate': .9}}))
    assert pilot_report(tmp_path)['selected'] == 'P3'
    (tmp_path / 'design.json').write_text(dumps({
        'product_regression_metrics': [],
        'minimum_product_metrics': {'exact_request_rate': .91}}))
    assert pilot_report(tmp_path)['selected'] == 'B1'


def test_bootstrap_keeps_multiple_requests_in_one_source_cluster(tmp_path):
    from tractor.training_report import clustered_difference

    before, after = [], []
    for ident, group, correct in [('a', 'shared', 1), ('b', 'shared', 0), ('c', 'other', 0)]:
        before.append({'id': ident, 'source_group': group, 'counts': {'requests': 1, 'exact_requests': 0}})
        after.append({'id': ident, 'source_group': group, 'counts': {'requests': 1, 'exact_requests': correct}})
    a, b = tmp_path / 'before.jsonl', tmp_path / 'after.jsonl'
    a.write_text(''.join(dumps(r) + '\n' for r in before))
    b.write_text(''.join(dumps(r) + '\n' for r in after))
    result = clustered_difference(a, b)
    assert result['source_groups'] == 2
    assert result['interval'] == [0, .5]
