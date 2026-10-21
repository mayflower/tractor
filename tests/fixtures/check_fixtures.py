#!/usr/bin/env python3
"""Validate API and normalization fixtures only; does not run a model or production parser."""
from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
UNITS = {
    'mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mile', 'nmi',
    'mg', 'g', 'kg', 'lb', 'ms', 's', 'min', 'h', 'mL', 'L', 'm3',
    'mm2', 'cm2', 'm2', 'm/s', 'km/h', 'mph',
}
ISSUES = {
    'not_found', 'ambiguous_value', 'ambiguous_record',
    'ambiguous_number_format', 'invalid_number_format', 'missing_unit',
    'ambiguous_unit', 'unsupported_unit', 'incompatible_dimension',
    'non_scalar_quantity', 'unsupported_number_format', 'numeric_out_of_range',
    'invalid_span',
}


def load(name: str) -> list[dict[str, Any]]:
    path = ROOT / name
    rows = []
    for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            raise AssertionError(f'{path.name}:{lineno}: invalid JSON') from exc
        rows.append(row)
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), f'{name}: duplicate IDs'
    return rows


def check_span(text: str, span: dict[str, Any]) -> None:
    start, end = span['start'], span['end']
    assert isinstance(start, int) and isinstance(end, int)
    assert 0 <= start < end <= len(text)
    assert text[start:end] == span['text'], (text, span)


def check_normalization() -> int:
    rows = load('normalization.jsonl')
    for row in rows:
        check_span(row['text'], row['span'])
        assert row['target_unit'] in UNITS
        expected = row['expected']
        if 'issue' in expected:
            assert expected['issue'] in ISSUES
            assert 'value_decimal' not in expected
            continue
        assert expected['unit'] == row['target_unit']
        value = Decimal(expected['value_decimal'])
        assert value.is_finite()
        if 'oracle' in row:
            oracle = row['oracle']
            with localcontext() as ctx:
                ctx.prec = 80
                calculated = Decimal(oracle['input_decimal']) * Decimal(oracle['factor_decimal'])
            assert value == calculated, (row['id'], value, calculated)
        if row['span']['text'] == '72 km/h':
            assert value == Decimal(72) * Decimal(1000) / Decimal(3600)
        if row['span']['text'] == '1 km/h':
            with localcontext() as ctx:
                ctx.prec = 80
                exact_reference = Decimal(5) / Decimal(18)
                assert abs(value - exact_reference) <= Decimal(expected['absolute_tolerance'])
    return len(rows)


def check_api() -> int:
    rows = load('api-cases.jsonl')
    for row in rows:
        assert row['origin'] == 'authored'
        assert row['source_group']
        request, gold = row['request'], row['gold']
        fields = request['fields']
        assert fields
        for spec in fields.values():
            assert spec['type'] in {'text', 'quantity'}
            if spec['type'] == 'quantity':
                assert spec['unit'] in UNITS
        many = request.get('many', False)
        anchor = request.get('anchor')
        if many:
            assert anchor in fields and fields[anchor]['type'] == 'text'
        for record in gold['records']:
            assert set(record) == set(fields)
            for spans in record.values():
                assert isinstance(spans, list)
                for span in spans:
                    check_span(request['text'], span)
            if anchor:
                assert len(record[anchor]) == 1
        data = gold['response']['data']
        assert isinstance(data, list) if many else isinstance(data, dict)
        out_records = data if many else [data]
        for index, record in enumerate(out_records):
            assert set(record) == set(fields)
            for name, value in record.items():
                if value is None:
                    continue
                candidates = value if isinstance(value, list) else [value]
                if isinstance(value, list):
                    assert len(value) > 1 and len(set(value)) == len(value)
                    assert any(issue['code'] == 'ambiguous_value' and issue['field'] == name
                               and issue.get('index', 0) == index for issue in gold['response']['issues'])
                for candidate in candidates:
                    if fields[name]['type'] == 'text':
                        assert isinstance(candidate, str)
                    else:
                        assert isinstance(candidate, (int, Decimal)) and not isinstance(candidate, bool)
                        assert Decimal(candidate).is_finite()
        for issue in gold['response']['issues']:
            assert issue['code'] in ISSUES
            name = issue['field']
            assert name is None or name in fields
            if 'index' in issue:
                assert many and 0 <= issue['index'] < len(data)
                value = data[issue['index']][name]
                assert isinstance(value, list) and len(value) > 1 if issue['code'] == 'ambiguous_value' else value is None
            elif name is not None and not many:
                value = data[name]
                assert isinstance(value, list) and len(value) > 1 if issue['code'] == 'ambiguous_value' else value is None
    # Independent arithmetic check for the two locale-variant regression rows.
    assert rows[6]['gold']['response']['data']['range_km'] == Decimal('1.234') * Decimal('1.609344')
    assert rows[7]['gold']['response']['data']['range_km'] == Decimal('1234') * Decimal('1.609344')
    return len(rows)


def main() -> None:
    normal_count = check_normalization()
    api_count = check_api()
    error_rows = load('http-errors.jsonl')
    for row in error_rows:
        assert row['expected_status'] == 422
        assert ('request' in row) != ('raw_body' in row)
    print(f'OK: {normal_count} normalization fixtures; {api_count} API fixtures; '
          f'{len(error_rows)} HTTP-error fixtures.')
    print('Checked: JSON structure, IDs, types, original character offsets and '
          'independent arithmetic oracles. No model inference or training executed.')


if __name__ == '__main__':
    main()
