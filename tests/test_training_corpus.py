from pathlib import Path

import pytest

from tractor.coverage_assembly import sha
from tractor.jsonio import dumps, loads
from tractor.training_corpus import assemble, verify_training_binding


def episode(ident, language='en', split='train', text=None):
    return {
        'id': ident, 'split': split, 'language': language, 'text': text or ident,
        'source_group': ident, 'derivation_group': ident, 'source': 'test-source',
        'text_origin': 'real', 'label_origin': 'human', 'family': 'contacts',
        'tasks': [{'id': 'document', 'kind': 'entities', 'fields': {'name': 'Name'}}],
        'gold': {'document': {'records': []}}, 'processing': {},
    }


def frozen(path, rows):
    path.write_text(''.join(dumps(r) + '\n' for r in rows))
    return {'path': str(path), 'sha256': sha(path)}


def configuration(tmp_path, groups, heldout=None):
    specs = [frozen(tmp_path / f'source-{i}.jsonl', rows) for i, rows in enumerate(groups)]
    dev = frozen(tmp_path / 'dev.jsonl', heldout or [episode('heldout', split='dev')])
    path = tmp_path / 'config.json'
    path.write_text(dumps({'languages': ['de', 'en'], 'sources': specs,
                           'heldout': [dev], 'dev': dev['path']}))
    return path


def test_full_selection_keeps_every_language_source_and_schema_without_quotas(tmp_path):
    rows = [episode(f'english-{i}') for i in range(257)] + [
        episode(f'german-{i}', 'de') for i in range(193)]
    config = configuration(tmp_path, [rows, [episode('supplement', 'de')]])
    result = assemble(config, tmp_path / 'out')
    actual = [loads(line) for line in (tmp_path / 'out/train.episodes.jsonl').open()]
    assert {r['id'] for r in actual} == {r['id'] for r in rows} | {'supplement'}
    assert result['languages'] == {'en': 257, 'de': 194}
    assert result['unique_train_episodes'] == 451
    assert result['training_ready'] is False  # Native targets still need validation.


def test_duplicate_exposures_preserve_preferred_annotation_and_audit_difference(tmp_path):
    corrected = episode('same')
    corrected['processing'] = {'document': {'name': {'kind': 'person'}}}
    config = configuration(tmp_path, [[corrected, corrected], [episode('same')]])
    result = assemble(config, tmp_path / 'out')
    actual = loads((tmp_path / 'out/train.episodes.jsonl').read_text())
    assert actual == corrected
    assert result['counts']['duplicate_ids'] == 2
    assert result['counts']['different_versions'] == 1


def test_non_target_languages_and_all_heldout_identity_forms_are_accounted(tmp_path):
    group_collision = episode('other-id')
    group_collision['derivation_group'] = 'heldout'
    rows = [episode('keep'), episode('indonesian', 'id'), episode('heldout'),
            episode('same-text', text='heldout'), group_collision]
    result = assemble(configuration(tmp_path, [rows]), tmp_path / 'out')
    assert result['unique_train_episodes'] == 1
    assert sum(v for k, v in result['counts'].items() if k.startswith('excluded:')) == 4
    assert len((tmp_path / 'out/selection.jsonl').read_text().splitlines()) == 5


def test_configuration_refuses_sampling_options(tmp_path):
    path = configuration(tmp_path, [[episode('example')]])
    config = loads(path.read_text())
    config['limit'] = 100
    path.write_text(dumps(config))
    with pytest.raises(ValueError, match='sampling'):
        assemble(path, tmp_path / 'out')


def test_source_hash_changes_abort_instead_of_silently_reducing_corpus(tmp_path):
    path = configuration(tmp_path, [[episode('example')]])
    source = Path(loads(path.read_text())['sources'][0]['path'])
    source.write_text('')
    with pytest.raises(ValueError, match='source changed'):
        assemble(path, tmp_path / 'out')


def test_changed_text_under_same_id_is_not_treated_as_duplicate(tmp_path):
    config = configuration(tmp_path, [[episode('same')], [episode('same', text='different')]])
    with pytest.raises(ValueError, match='Conflicting identity'):
        assemble(config, tmp_path / 'out')


def test_nontraining_input_fails_even_when_language_matches(tmp_path):
    config = configuration(tmp_path, [[episode('not-train', split='test')]])
    with pytest.raises(ValueError, match='Non-training row'):
        assemble(config, tmp_path / 'out')


def test_training_binding_rejects_substituted_subset_and_modified_native_file(tmp_path):
    full = tmp_path / 'full'
    full.mkdir()
    names = ['train.native.jsonl', 'train.identities.jsonl', 'dev.episodes.jsonl', 'dev.native.jsonl']
    for name in names:
        (full / name).write_text('{"example": 1}\n')
    manifest = {'full_corpus': True, 'training_ready': True,
                'unique_train_episodes': 1, 'native_requests': {'train': 1},
                'files': {name: sha(full / name) for name in names}}
    (full / 'manifest.json').write_text(dumps(manifest))
    binding = {'path': str(full), 'manifest_sha256': sha(full / 'manifest.json'),
               'unique_train_episodes': 1, 'native_train_requests': 1}
    assert verify_training_binding(full, binding) == manifest
    with pytest.raises(ValueError, match='differs'):
        verify_training_binding(tmp_path / 'subset', binding)
    (full / 'train.native.jsonl').write_text('')
    with pytest.raises(ValueError, match='artifact changed'):
        verify_training_binding(full, binding)
