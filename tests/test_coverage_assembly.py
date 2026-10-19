import pytest

from tractor.coverage_assembly import assemble, sha
from tractor.jsonio import dumps, loads
from tractor.synthetic_coverage import make_episode


def sources(tmp_path):
    base, synth = tmp_path / 'base', tmp_path / 'synth'
    for root in (base, synth):
        root.mkdir()
        (root / 'supplement').mkdir()
        files = {}
        for split in ('train', 'dev', 'calibration', 'test'):
            directory = root / 'supplement' if root == base and split in ('calibration', 'test') else root
            episode, _ = make_episode(split, 10 if root == base else 20, 'document_fields', 'de')
            for kind, value in [('episodes', episode.model_dump()), ('native', {'input': episode.text, 'output': {}})]:
                path = directory / f'{split}.{kind}.jsonl'
                path.write_text(dumps(value) + '\n')
                files[str(path.relative_to(root))] = sha(path)
        manifest = {'status': 'complete' if root == base else 'validated',
                    'scale_status': 'passed_matched_dev_pilot', 'files': files}
        (root / 'manifest.json').write_text(dumps(manifest))
    return base, synth


def test_assembly_preserves_counts_and_paired_texts(tmp_path):
    base, synth = sources(tmp_path)
    report = assemble(base, synth, tmp_path / 'combined')
    assert report['counts'] == dict.fromkeys(('train', 'dev', 'calibration', 'test'), 2)
    assert report['unique_texts'] == 8


def test_changed_source_is_refused_before_assembly(tmp_path):
    base, synth = sources(tmp_path)
    (synth / 'train.native.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Unverified'):
        assemble(base, synth, tmp_path / 'combined')
    assert not (tmp_path / 'combined').exists()


def test_scale_gate_cannot_be_skipped(tmp_path):
    base, synth = sources(tmp_path)
    path = synth / 'manifest.json'
    data = loads(path.read_text())
    data['scale_status'] = 'blocked_pending_matched_dev_pilot'
    path.write_text(dumps(data))
    with pytest.raises(ValueError, match='gated'):
        assemble(base, synth, tmp_path / 'combined')


def test_cross_split_group_cannot_be_hidden_in_source_manifests(tmp_path):
    base, synth = sources(tmp_path)
    train = loads((base / 'train.episodes.jsonl').read_text())
    path = synth / 'dev.episodes.jsonl'
    dev = loads(path.read_text())
    dev['source_group'] = train['source_group']
    path.write_text(dumps(dev) + '\n')
    mp = synth / 'manifest.json'
    m = loads(mp.read_text())
    m['files'][path.name] = sha(path)
    mp.write_text(dumps(m))
    with pytest.raises(ValueError, match='Source group'):
        assemble(base, synth, tmp_path / 'combined')
    assert not (tmp_path / 'combined/manifest.json').exists()


def test_classification_supplement_is_appended_with_verified_hashes(tmp_path):
    base, synth = sources(tmp_path)
    cls = tmp_path / 'classification'
    cls.mkdir()
    files = {}
    for split in ('train', 'dev', 'calibration', 'test'):
        e, _ = make_episode(split, 30, 'document_fields', 'en')
        for kind, value in [('episodes', e.model_dump()), ('native', {'input': e.text, 'output': {}})]:
            p = cls / f'{split}.{kind}.jsonl'
            p.write_text(dumps(value) + '\n')
            files[p.name] = sha(p)
    (cls / 'manifest.json').write_text(dumps({'status': 'validated', 'files': files}))
    result = assemble(base, synth, tmp_path / 'combined', classification=cls)
    assert result['counts'] == dict.fromkeys(('train', 'dev', 'calibration', 'test'), 3)
    (cls / 'train.native.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Unverified'):
        assemble(base, synth, tmp_path / 'tampered', classification=cls)
