"""Offline reference adequacy; never relax the production extraction contract."""

VERSION = 'reviewed-text-equivalence-v1'


def incomplete_reference(episode, task, key, answer):
    """A cell's short name is not exhaustive gold for a broad page-level request.

    Applies to every split/site/checkpoint, including renamed fields. No
    predictions or preferred answer boundaries enter this policy.
    """
    if not answer.reference_complete:
        return True
    rule = episode.processing.get(task.id, {}).get(key)
    return (
        episode.source == 'swde'
        and episode.source_group.split(':')[1] in {'auto', 'camera'}
        and task.kind == 'entities'
        and task.fields[key] in {
            'auto model explicitly written on the page',
            'camera model explicitly written on the page',
        }
        and (rule is None or (
            rule.kind == 'literal' and rule.cardinality in {'unique_list', 'one_or_more'}
        ))
    )


def comparison(value, answer, *, open_reference=False, key=lambda value: value):
    """Only declared equivalents, never substring/fuzzy similarity or inferred aliases."""
    def normalized(value):
        value = key(value)
        return ' '.join(value.split()) if isinstance(value, str) else value
    if not (open_reference or answer.value_equivalences):
        return key(value)
    value = normalized(value)
    for group in answer.value_equivalences:
        members = [normalized(v) for v in group]
        if value in members:
            return members[0]
    return value
