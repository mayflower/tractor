from collections import Counter

import pytest

from tractor.episodes import audit_episodes
from tractor.synthetic_coverage import (
    FAMILIES,
    SPLITS,
    Document,
    allocation,
    generate_split,
    make_episode,
)


def test_repeated_values_keep_occurrence_identity():
    doc = Document()
    a = doc.render('{head} owns {tail}. ', {'head': 'Delta', 'tail': 'Delta'})
    assert a['head']['items'][0]['start'] == 0
    assert a['tail']['items'][0]['start'] == 11
    assert a['head']['items'] != a['tail']['items']


@pytest.mark.parametrize('split', SPLITS)
@pytest.mark.parametrize('family', FAMILIES)
def test_evidence_and_complete_targets(split, family):
    episode, meta = make_episode(split, 12, family, 'de')
    assert episode.text_origin == 'synthetic'
    assert meta['render_family'].endswith(split)
    for gold in episode.gold.values():
        assert gold.complete
        for record in gold.records:
            for answer in record.values():
                for span in answer.items:
                    assert episode.text[span.start:span.end] == span.text


def test_contrast_pair_is_grouped_and_changes_answer():
    positive, _ = make_episode('dev', 44, 'role_contrasts', 'en', variant=0)
    negative, _ = make_episode('dev', 44, 'role_contrasts', 'en', variant=1)
    assert positive.source_group == negative.source_group
    assert positive.text != negative.text
    assert positive.gold['selection'].records[0]['value'].status == 'present'
    assert negative.gold['selection'].records[0]['value'].status == 'absent'


def test_allocation_and_split_independence():
    config = {'seed': 731, 'family_weights': dict(zip(FAMILIES, [25, 20, 15, 15, 15, 10], strict=True)),
              'language_weights': {'de': 45, 'en': 45, 'mixed': 10}}
    episodes = []
    for split in SPLITS:
        rows = list(generate_split(split, 1000, config))
        assert Counter(e.family for e, _ in rows) == {k: v * 10 for k, v in config['family_weights'].items()}
        episodes.extend(e for e, _ in rows)
    assert audit_episodes(episodes)['episodes'] == dict.fromkeys(SPLITS, 1000)
    with pytest.raises(ValueError):
        allocation(101, config['family_weights'])


def test_cross_section_targets_follow_reference_not_order():
    episode, _ = make_episode('train', 10, 'cross_section', 'en')
    for record in episode.gold['extract'].records:
        anchor = record['anchor'].items[0]
        amount = record['amount'].items[0]
        assert amount.start > episode.text.index('Details by reference')
        detail_line = episode.text[:amount.start].splitlines()[-1]
        assert detail_line.startswith(anchor.text + ': ')


def test_offset_blocks_do_not_reuse_scenario_ids():
    config = {'seed': 731, 'family_weights': dict(zip(FAMILIES, [25, 20, 15, 15, 15, 10], strict=True)),
              'language_weights': {'de': 45, 'en': 45, 'mixed': 10}}
    a = list(generate_split('train', 1000, config))
    b = list(generate_split('train', 1000, config, offset=1000))
    assert not {e.id for e, _ in a} & {e.id for e, _ in b}
    assert Counter(e.language for e, _ in a) == {'de': 450, 'en': 450, 'mixed': 100}


def test_scale_cannot_bypass_failed_product_or_public_gate():
    from copy import deepcopy

    from tractor.synthetic_coverage import require_scale_gate

    config = {'scale_requires': {'synthetic_macro_f1_improvement': .02,
                                 'public_macro_f1_max_drop': .01, 'product_exact_request_floor': .90}}
    report = {'status': 'complete', 'scale_allowed': True, 'selected': 'S20', 'scores': {
        'S0': {'synthetic_macro_f1': .5, 'public_macro_f1': .6, 'product_exact_request': .95},
        'S20': {'synthetic_macro_f1': .6, 'public_macro_f1': .6, 'product_exact_request': .92}}}
    require_scale_gate(report, config)
    for key, value in [('product_exact_request', .899), ('public_macro_f1', .58),
                       ('synthetic_macro_f1', .51), ('synthetic_macro_f1', float('nan'))]:
        bad = deepcopy(report)
        bad['scores']['S20'][key] = value
        with pytest.raises(ValueError):
            require_scale_gate(bad, config)


def test_scaling_preserves_frozen_pilot_and_avoids_new_body_duplicates():
    from tractor.synthetic_coverage import training_blocks

    config = {'seed': 731, 'pilot_train': 10000,
              'family_weights': dict(zip(FAMILIES, [25, 20, 15, 15, 15, 10], strict=True)),
              'language_weights': {'de': 45, 'en': 45, 'mixed': 10}}
    rows = list(training_blocks(config, 2))
    assert len(rows) == 20000
    original = list(generate_split('train', 10000, config))
    assert [e for e, _ in rows[:10000]] == [e for e, _ in original]
    seen = {' '.join(e.text.split('\n', 1)[1].split()).casefold() for e, _ in original}
    for episode, _ in rows[10000:]:
        body = ' '.join(episode.text.split('\n', 1)[1].split()).casefold()
        assert body not in seen
        seen.add(body)


@pytest.mark.parametrize('role', range(5))
def test_expanded_scalar_roles_keep_absence_explicit(role):
    index = 10000 + 2 * role
    present, _ = make_episode('train', index, 'role_contrasts', 'en', variant=0)
    absent, _ = make_episode('train', index, 'role_contrasts', 'en', variant=1)
    answer = present.gold['selection'].records[0]['value']
    assert answer.status == 'present'
    assert present.text[answer.items[0].start:answer.items[0].end] == answer.items[0].text
    assert absent.gold['selection'].records[0]['value'].status == 'absent'
    assert 'No value is provided' in absent.text
    assert present.source_group == absent.source_group


def test_scale_records_include_explicitly_missing_optional_fields():
    episode, meta = make_episode('train', 10000, 'records', 'en')
    rows = episode.gold['extract'].records
    assert 'missing_record_field' in meta['difficulties']
    assert any(a.status == 'absent' for row in rows for a in row.values())
    assert all(row['anchor'].status == 'present' for row in rows)


def test_scale_replaces_whole_overlapping_pair_and_preserves_pilot():
    from tractor.synthetic_coverage import training_blocks

    config = {'seed': 731, 'pilot_train': 100, 'family_weights': {'role_contrasts': 100},
              'language_weights': {'en': 100}}
    frozen = list(training_blocks(config, 1))

    class Protected:
        def overlaps(self, episode):
            return 'protected_lexical_containment' if episode.id.endswith(':100:1') else None

    replacements = []
    rows = list(training_blocks(config, 2, Protected(), replacements))
    assert rows[:100] == frozen
    assert len(rows) == 200
    assert not any(e.id.endswith((':100:0', ':100:1')) for e, _ in rows)
    assert rows[100][0].source_group == rows[101][0].source_group
    assert replacements == [{'id': 'coverage:731:train:100:1', 'reason': 'protected_lexical_containment'}]


def test_protected_overlap_never_rewrites_frozen_pilot():
    from tractor.synthetic_coverage import training_blocks

    class Protected:
        def overlaps(self, episode):
            return 'protected_source_group'

    config = {'seed': 731, 'pilot_train': 100, 'family_weights': {'role_contrasts': 100},
              'language_weights': {'en': 100}}
    with pytest.raises(ValueError, match='Frozen pilot'):
        list(training_blocks(config, 1, Protected()))
