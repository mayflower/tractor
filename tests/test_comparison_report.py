from decimal import Decimal

import pytest

from tractor.comparison_report import rounded_equal, summarize, wilson


def test_tolerance_preserves_extraction_contract():
    assert rounded_equal({'width': Decimal('1.0000005')}, {'width': 1})
    assert rounded_equal(Decimal('0.0000005'), 0)
    assert not rounded_equal(Decimal('1.000002'), 1)
    assert not rounded_equal('1', 1)
    assert not rounded_equal(True, 1)
    assert not rounded_equal({'width': 1}, {'length': 1})
    assert not rounded_equal([1, 2], [2, 1])
    assert not rounded_equal(None, 0)


def test_failed_calls_stay_in_denominator_and_latency():
    gold = {'response': {'data': {'width': 1}, 'issues': []}}
    rows = {str(i): {'scenario': 'width', 'gold': gold} for i in range(3)}
    records = [
        {'id': '0', 'latency_ms': 10, 'http_status': 200,
         'result': {'data': {'width': 1}, 'issues': []}},
        {'id': '1', 'latency_ms': 20, 'http_status': 200,
         'result': {'data': {'width': Decimal('1.0000005')}, 'issues': []}},
        {'id': '2', 'latency_ms': 100, 'http_status': 429, 'error': 'HTTPStatusError'},
    ]
    result = summarize(rows, records)
    assert result['counts']['exact'] == 1
    assert result['counts']['rounded_exact'] == 2
    assert result['counts']['errors'] == 1
    assert result['rates']['rounded_data_exact'] == 2/3
    assert result['p95_ms'] == 100
    assert result['valid_response_p95_ms'] == 20
    with pytest.raises(ValueError):
        summarize(rows, records[:2])
    with pytest.raises(ValueError):
        summarize(rows, records[:2] + records[:1])
    lower, upper = wilson(50, 100)
    assert .40 < lower < .41
    assert .59 < upper < .60


def test_public_evidence_can_be_rescored_without_credentials(tmp_path, monkeypatch):
    import hashlib
    import sys

    from tractor.comparison_report import main
    from tractor.jsonio import dumps, loads

    data = tmp_path/'data.jsonl'
    result = {'data': {'width': 1}, 'issues': []}
    data.write_text(dumps({'id': 'one', 'scenario': 'width', 'gold': {'response': result}})+'\n')
    manifest = {'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
                'limit': 1, 'ids': ['one'], 'prompt_sha256': 'prompt', 'runner_sha256': 'runner'}
    args = ['report', '--data', str(data)]
    for name in ('tractor', 'haiku', 'luna'):
        directory = tmp_path/name
        directory.mkdir()
        (directory/'manifest.json').write_text(dumps(manifest))
        record = {'id': 'one', 'latency_ms': 10, 'http_status': 200,
                  'result': result, 'request_id': 'private-request-id'}
        if name != 'tractor':
            record['raw_response'] = {'model': name, 'usage': {
                'input_tokens': 100, 'output_tokens': 10},
                'content': [{'type': 'text', 'text': dumps(result)}],
                'output': [{'type': 'message', 'content': [
                    {'type': 'output_text', 'text': dumps(result)}]}]}
        (directory/'responses.jsonl').write_text(dumps(record)+'\n')
        args += ['--'+name, str(directory)]
    output = tmp_path/'public'
    monkeypatch.setattr(sys, 'argv', args+['--output', str(output)])
    main()
    first = loads((output/'summary.json').read_text())
    assert first['providers']['haiku']['counts']['exact'] == 1
    assert first['providers']['luna']['estimated_api_cost_usd'] == Decimal('0.000015')
    public_text = (output/'haiku/responses.jsonl').read_text()
    assert 'private-request-id' not in public_text
    assert loads(public_text)['provider_output'] == dumps(result)
    rescored = tmp_path/'rescored'
    args = ['report', '--data', str(data), '--output', str(rescored)]
    for name in ('tractor', 'haiku', 'luna'):
        args += ['--'+name, str(output/name)]
    monkeypatch.setattr(sys, 'argv', args)
    main()
    assert loads((rescored/'summary.json').read_text()) == first
