from tractor.coverage_assembly import sha
from tractor.document_sources import envelope
from tractor.jsonio import dumps, loads
from tractor.typed_corpus import build
from tractor.typed_mixture import attach


def test_attach_preserves_exposure_order_repetition_and_native_bytes(tmp_path):
    entries, roots = [], []
    for name, text in [('public', 'Jane Doe'), ('authored', 'Max Müller')]:
        source, target = tmp_path / name, tmp_path / (name + '-typed')
        source.mkdir()
        row = envelope('tractor_authored_documents', name, 'train', 'research', 'en', text,
            [{'id': 'doc', 'kind': 'entities', 'description': 'Author', 'fields': {'author': 'Named author'}}],
            {'doc': {'complete': True, 'records': [{'author': {'status': 'present',
                'items': [{'start': 0, 'end': len(text), 'text': text}]}}]}})
        (source / 'train.episodes.jsonl').write_text(dumps(row.model_dump())+'\n')
        (source / 'train.native.jsonl').write_text(dumps({'input': text})+'\n')
        files = {p.name: sha(p) for p in source.iterdir()}
        (source / 'manifest.json').write_text(dumps({'files': files, 'training_ready': True}))
        build(source, target)
        entries.append(row)
        roots.append(target)
    mixture = tmp_path / 'mixture'
    mixture.mkdir()
    schedule = [entries[1], entries[0], entries[1]]
    (mixture / 'train.identities.jsonl').write_text(''.join(
        dumps({'id': row.id, 'source_group': row.source_group})+'\n' for row in schedule))
    (mixture / 'train.native.jsonl').write_text(''.join(dumps({'input': row.text})+'\n' for row in schedule))
    files = {p.name: sha(p) for p in mixture.iterdir()}
    (mixture / 'manifest.json').write_text(dumps({'files': files, 'type_migration': {'test': True}}))
    report = attach(mixture, roots[1], roots[0])
    assert report['exposures'] == 3 and report['unique_episodes'] == 2
    migrated = [loads(line) for line in (mixture / 'train.episodes.jsonl').open()]
    assert [row['id'] for row in migrated] == [row.id for row in schedule]
    assert all(row['processing']['doc']['author']['kind'] == 'person' for row in migrated)
    assert sha(mixture / 'train.native.jsonl') == files['train.native.jsonl']
    assert sha(mixture / 'train.identities.jsonl') == files['train.identities.jsonl']
