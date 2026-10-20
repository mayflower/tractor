"""Frozen corpus-to-public-type policy. Reads schemas/metadata/text, never answers.

Names shared by different source schemas are intentionally not interchangeable.
Unspecified output parameters and mixed semantic fields remain explicit debt.
"""
import re

from .record_contracts import cardinality

VERSION = 'typed-corpus-20261019-v2-cardinality'

PERSON = {
    'research': {'author'}, 'clinical': {'patient', 'clinician'},
    'communication': {'contact'}, 'contacts': {'name'}, 'forms': {'applicant', 'name'},
    'news': {'person'}, 'operations': {'technician'},
}
ORGANIZATION = {
    'invoices': {'supplier', 'customer'}, 'contacts': {'company'},
    'careers': {'employer', 'company'}, 'financial_reports': {'company', 'auditor'},
    'forms': {'authority'}, 'banking': {'employer'}, 'news': {'company'},
    'metadata': {'charity_name'}, 'research': {'publisher'}, 'products': {'manufacturer'},
}
PLACE = {'careers': {'location'}, 'news': {'location'}, 'operations': {'location'},
         'logistics': {'origin', 'destination'}, 'contracts': {'jurisdiction'}}
IDENTIFIERS = {
    'invoices': {'number', 'code'}, 'products': {'code'},
    'forms': {'reference'}, 'clinical': {'reference'},
    'metadata': {'number', 'case', 'reference', 'charity_number'},
    'banking': {'account', 'identity', 'statement'},
    'communication': {'customer', 'order', 'contract', 'reference'},
    'contracts': {'reference'}, 'insurance': {'policy', 'claim'},
    'logistics': {'container', 'shipment', 'customs_code'},
    'news': {'ticker'}, 'operations': {'reference', 'asset'}, 'research': {'isbn_13', 'sample'},
}
MONEY = {'invoices': {'net', 'tax', 'gross', 'unit_price', 'line_total'},
         'news': {'amount'}, 'forms': {'amount'}, 'banking': {'income', 'amount'},
         'insurance': {'coverage', 'deductible', 'amount'},
         'contracts': {'fee', 'amount'}, 'products': {'price'}, 'careers': {'salary'}}
DATE = {'date', 'birth_date', 'start', 'end', 'submitted', 'due', 'published',
        'followup', 'report_date', 'booking_date', 'publication_date', 'date_posted'}
COUNTS = {'contacts': {'employees'}, 'research': {'sample_size'},
          'communication': {'quantity'}, 'invoices': {'quantity'}, 'logistics': {'packages'}}


def currency_context(text):
    """Only one explicitly named currency in the whole document is usable.

    A bare dollar sign does not specify USD/CAD/AUD. Language is not currency.
    This decision is fixed before inspecting spans, including absent fields.
    """
    found = set()
    for code, pattern in {
        'EUR': r'\bEUR\b|\bEuros?\b|€', 'GBP': r'\bGBP\b|pounds? sterling|£',
        'USD': r'\bUSD\b|\bUS dollars?\b|US\$',
        'CAD': r'\bCAD\b|Canadian dollars?|CA\$',
        'AUD': r'\bAUD\b|Australian dollars?|AU\$',
        'CHF': r'\bCHF\b|Swiss francs?|Schweizer Franken',
    }.items():
        if re.search(pattern, text, re.I):
            found.add(code)
    if '$' in text and not found & {'USD', 'CAD', 'AUD'}:
        return None
    return next(iter(found)) if len(found) == 1 else None


def choose(schema, task, key, existing=None):
    """Return (rule, debt). schema contains no gold, ids, split or predictions."""
    rule, debt = _choose(schema, task, key, existing)
    rule = dict(rule, cardinality=cardinality(task.get('cardinalities', {}).get(key), rule.get('cardinality')))
    return rule, debt


def _choose(schema, task, key, existing=None):
    source, family = schema['source'], schema['family']
    kind, options, debt = None, {}, None
    authored = source in {'tractor_authored_documents', 'tractor_authored_gb_offices'}
    if source == 'few_nerd':
        kind = {'person': 'person', 'organization': 'organization', 'location': 'place',
                'building': 'place'}.get(key)
    elif source == 'docred':
        kind = {'PER': 'person', 'ORG': 'organization', 'LOC': 'place', 'NUM': 'number'}.get(key)
        if key == 'TIME':
            debt = 'mixed_date_time_and_duration_schema'
    elif source == 'swde':
        vertical = schema['source_group'].split(':')[1]
        kind = {
            ('nbaplayer', 'name'): 'person', ('nbaplayer', 'team'): 'organization',
            ('restaurant', 'name'): 'organization', ('university', 'name'): 'organization',
            ('restaurant', 'address'): 'address', ('job', 'company'): 'organization',
            ('job', 'location'): 'place', ('job', 'date_posted'): 'date',
            ('book', 'author'): 'person', ('book', 'publisher'): 'organization',
            ('book', 'isbn_13'): 'identifier', ('book', 'publication_date'): 'date',
            ('camera', 'manufacturer'): 'organization', ('university', 'website'): 'url',
            ('restaurant', 'phone'): 'phone', ('university', 'phone'): 'phone',
            ('movie', 'director'): 'list', ('auto', 'price'): 'money', ('camera', 'price'): 'money',
        }.get((vertical, key))
        if key in {'height', 'weight'}:
            debt = 'source_schema_has_no_explicit_target_unit'
        if kind == 'list':
            options['separator'] = 'newline'
    elif source == 'kleister-charity':
        if key == 'charity_name':
            kind = 'organization'
        elif key == 'charity_number':
            kind = 'identifier'
        elif key.startswith('address__'):
            component = {'address__post_town': 'city', 'address__postcode': 'postcode',
                         'address__street_line': 'street'}[key]
            kind, options = 'address', dict(country='GB', component=component, group='registered_office')
    elif source == 'kleister-nda':
        kind = {'jurisdiction': 'place', 'term': 'duration'}.get(key)
        if key == 'party':
            debt = 'party_schema_allows_person_or_organization'
    elif source == 'vrdu':
        kind = {'registration_num': 'identifier', 'property': 'organization', 'gross_amount': 'money'}.get(key)
        if key == 'advertiser':
            debt = 'advertiser_schema_allows_person_or_organization'
    elif source == 'cord':
        debt = 'source_contract_requests_entire_printed_cell_including_label'
    if authored:
        if key in PERSON.get(family, set()):
            kind = 'person'
        elif key in ORGANIZATION.get(family, set()):
            kind = 'organization'
        elif key in PLACE.get(family, set()):
            kind = 'place'
        elif key in IDENTIFIERS.get(family, set()):
            kind = 'identifier'
        elif key in MONEY.get(family, set()):
            kind = 'money'
        elif key in COUNTS.get(family, set()):
            kind = 'number'
        elif key in DATE:
            kind = 'datetime' if family == 'operations' and key == 'date' else 'date'
        elif key == 'address':
            kind = 'address'
        elif key in {'phone', 'callback'}:
            kind = 'phone'
        elif key == 'email':
            kind = 'email'
        elif key == 'website':
            kind = 'url'
        elif key == 'iban':
            kind = 'iban'
        elif family == 'banking' and key == 'period':
            kind = 'interval'
        elif family == 'contracts' and key == 'term':
            kind = 'duration'
        elif family == 'contracts' and key == 'notice':
            debt = 'notice_clause_includes_calendar_conditions_not_just_duration'
        elif key == 'weight' and family in {'products', 'logistics'}:
            kind, options = 'quantity', {'unit': 'kg'}
        elif family == 'products' and key == 'length':
            kind, options = 'quantity', {'unit': 'm'}
        elif (family, key) in {('careers', 'name'), ('news', 'name'), ('insurance', 'insured'),
                               ('banking', 'customer'), ('banking', 'counterparty'),
                               ('contracts', 'party_a'), ('contracts', 'party_b'),
                               ('metadata', 'sender'), ('metadata', 'recipient'),
                               ('logistics', 'sender'), ('logistics', 'recipient')}:
            debt = 'mixed_name_or_composite_contact_schema'
        elif (family, key) in {('research', 'value'), ('operations', 'measurement'),
                               ('operations', 'limit'), ('clinical', 'dose'), ('insurance', 'quantity')}:
            debt = 'heterogeneous_dimensions_no_declared_target_unit'
        elif (family, key) in {('communication', 'appointment'), ('financial_reports', 'period'),
                               ('operations', 'time')}:
            debt = 'mixed_temporal_precision_or_point_and_interval_schema'
        elif family == 'financial_reports' and key == 'value':
            kind = 'number'  # Printed scalar; unit stays a separate explicit field.
    if kind == 'money':
        currency = existing.get('currency') if existing and existing.get('kind') == 'money' else currency_context(schema['text'])
        if currency:
            options['currency'] = currency
        else:
            kind, debt = None, 'missing_or_multiple_explicit_currency'
    if kind == 'date':
        options['allow_partial'] = True
    if kind in {'date', 'datetime', 'interval'} and authored and schema['language'] == 'de':
        options['date_order'] = 'DMY'  # Existing authored German corpus convention.
    if kind in {'money', 'number', 'quantity', 'percent'} and schema.get('locale'):
        options['locale'] = schema['locale']
    if task.get('anchor') == key and kind not in {None, 'person', 'organization', 'place', 'identifier'}:
        kind, options, debt = None, {}, 'record_anchor_must_preserve_literal_identity'
    if existing and existing['kind'] != 'literal':
        # Preserve predeclared units, countries and cardinalities. The operations
        # schema explicitly requests date AND time, correcting that old date rule.
        if family == 'operations' and key == 'date' and kind == 'datetime':
            pass
        else:
            return existing, None
    if kind is None:
        return existing or {'kind': 'literal'}, debt
    if existing:
        for option in ('cardinality', 'prefixes'):
            if option in existing:
                options[option] = existing[option]
    if source == 'swde' and kind in {'phone', 'date'} and not options.get('prefixes'):
        labels = {'phone': ['Phone:', 'Telephone:', 'Phone Number:', ':'],
                  'publication_date': ['Pub. Date:', 'Publication Date:', ':']}
        if key in labels:
            options['prefixes'] = labels[key]
    options.setdefault('cardinality', {'zero_or_more': 'unique_list'}.get(
        task.get('cardinalities', {}).get(key), task.get('cardinalities', {}).get(key, 'unique_list')))
    return {'kind': kind, **options}, debt
