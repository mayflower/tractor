from tractor.coverage_assembly import sha
from tractor.document_quarantine import quarantine
from tractor.jsonio import dumps, loads


def test_quarantine_removes_whole_group_and_counts_sources_not_windows(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    files = {}
    for split in ('train', 'dev', 'calibration', 'test'):
        episodes = []
        if split == 'train':
            for ident, group in [('a', 'blocked'), ('b', 'blocked'), ('c', 'kept'), ('d', 'kept')]:
                episodes.append({'id': ident, 'source_group': group, 'source': 'source', 'label_origin': 'human',
                                 'split': split, 'text': ident, 'gold': {'task': {'records': [{'value': {'items': [{'text': ident}]}}]}}})
        ep = source / f'{split}.episodes.jsonl'
        np = source / f'{split}.native.jsonl'
        ep.write_text(''.join(dumps(row) + '\n' for row in episodes))
        np.write_text(''.join(dumps({'input': row['text']}) + '\n' for row in episodes))
        files.update({ep.name: sha(ep), np.name: sha(np)})
    (source / 'manifest.json').write_text(dumps({'files': files, 'qualification': 'test fixture'}))
    audit = tmp_path / 'audit.json'
    audit.write_text(dumps({'cross_split_groups_to_quarantine': ['blocked'],
                           'inputs': {k: v for k, v in files.items() if '.episodes.' in k}}))
    output = tmp_path / 'result'
    quarantine(source, audit, output)
    result = loads((output / 'manifest.json').read_text())
    assert result['unique_real_train'] == 1
    assert result['counts']['train'] == 2
    assert result['quarantine']['removed_windows']['train'] == 2
    assert not result['minimum_real_train_met'] and not result['training_ready']
    assert {loads(line)['id'] for line in (output / 'train.episodes.jsonl').read_text().splitlines()} == {'c', 'd'}
