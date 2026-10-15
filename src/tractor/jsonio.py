"""Decimal-preserving JSON with duplicate-key rejection."""

import simplejson


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def loads(value):
    return simplejson.loads(value, use_decimal=True, allow_nan=False, object_pairs_hook=_unique)


def dumps(value, **kwargs):
    return simplejson.dumps(value, use_decimal=True, allow_nan=False, ensure_ascii=False, **kwargs)
