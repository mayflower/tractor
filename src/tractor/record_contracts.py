"""Preserve declared value cardinalities through legacy corpus migration."""
from copy import deepcopy


def cardinality(declared=None, processing=None):
    declared = 'unique_list' if declared == 'zero_or_more' else declared
    processing = 'unique_list' if processing == 'zero_or_more' else processing
    # unique_list was the implicit Processing default, including serialized
    # defaults. It must not overwrite an independently declared task contract.
    if declared and processing not in (None, 'unique_list', declared):
        raise ValueError('Conflicting explicit field cardinalities')
    return declared or processing or 'unique_list'


def migrate(row):
    """Change contracts only; never infer cardinality from answers or predictions."""
    result = deepcopy(row)
    changes = []
    for task in result['tasks']:
        if task['kind'] != 'records':
            continue
        rules = result.setdefault('processing', {}).setdefault(task['id'], {})
        for key in task['fields']:
            rule = rules.setdefault(key, {'kind': 'literal'})
            declared = task.get('cardinalities', {}).get(key)
            before = rule.get('cardinality', 'unique_list')
            after = cardinality(declared, before)
            rule['cardinality'] = after
            if before != after:
                changes.append({'task': task['id'], 'field': key, 'before': before, 'after': after})
    return result, changes
