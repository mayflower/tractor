"""Permanent source exclusions, shared by corpus building and evaluation."""


def excluded_source(row):
    get = row.get if isinstance(row, dict) else lambda key, default: getattr(row, key, default)
    source = get('source', '').strip().lower()
    return source in {'cord', 'cord-v2', 'naver-clova-ix/cord-v2'} or any(
        str(get(key, '')).lower().startswith('cord:')
        for key in ('id', 'source_group', 'derivation_group')
    )
