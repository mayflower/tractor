import json
from collections import Counter

from tractor.pipeline_supplement import prepare


def test_focused_plan_is_bounded_training_only_with_explicit_prose_records(tmp_path):
    root = tmp_path / 'plan'
    prepare(root, focus=True)
    rows = [json.loads(line) for line in (root / 'plan.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 256
    assert Counter(r['domain'] for r in rows) == {'research': 160, 'careers': 48, 'contacts': 48}
    assert {r['split'] for r in rows} == {'train'}
    assert {r['language'] for r in rows} == {'de', 'en'}
    for row in rows:
        assert row['shape'] == 'repeated_records'
        assert 'No tables' in row['authorship_brief']
        record = next(t for t in row['tasks'] if t['kind'] == 'records')
        assert record['anchor'] in record['fields']
        assert 2 <= len(row['rows'][record['id']]) <= 4
