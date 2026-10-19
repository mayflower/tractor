"""Conservative local-evidence qualification; never invent or normalize targets."""
import re
from copy import deepcopy

FINANCIAL_FIELDS = {'income_annually_in_british_pounds', 'spending_annually_in_british_pounds'}
VRDU_LABELS = {
    'registration_num': r'registration\s*(?:no\.?|number)',
    'registrant_name': r'registrant\s*name',
    'foreign_principle_name': r'foreign\s*principal',
    'file_date': r'(?:date\s*signed|date\s*of\s*signature)',
    'signer_name': r'(?:signed\s*by|signer\s*name)',
    'signer_title': r'signer\s*title',
    'property': r'(?:property|station)', 'advertiser': r'advertiser',
    'agency': r'agency', 'product': r'product',
    'contract_num': r'(?:contract|order)\s*(?:#|no\.?|number)',
    'gross_amount': r'gross\s*(?:amount|total)',
    'flight_from': r'flight\s*(?:from|start)', 'flight_to': r'flight\s*(?:to|end)',
    'tv_address': r'station\s*address',
}


def explicit_local_role(text, key, span, source):
    prefix = text[max(0, span['start']-160):span['start']]
    if source == 'kleister-nda' and key == 'jurisdiction':
        # The location must follow an explicit governing-law role, not merely
        # occur in a document that happens to contain a governing-law clause.
        return bool(re.search(r'(?:laws?\s+of|courts?\s+of)\s+(?:the\s+)?(?:state\s+of\s+)?$', prefix, re.I))
    if source == 'kleister-charity' and key in FINANCIAL_FIELDS:
        label = r'(?:total|annual)\s+(?:income|receipts)' if key.startswith('income') else r'(?:total|annual)\s+(?:expenditure|expenses|spending)'
        return bool(re.search(label + r'\s*(?:(?:was|were|of|amounted to)\s*)?[:]?\s*[£$]?\s*$', prefix, re.I))
    if source == 'vrdu':
        label = VRDU_LABELS.get(key)
        if not label:
            return False
        # Same line or immediately following the label; no intervening column
        # names/values and no reconstruction using OCR coordinates.
        return bool(re.search(r'(?:^|\n)\s*'+label+r'\s*[:#]?\s*\n?\s*$', prefix, re.I))
    return True


def restrict(row):
    if row['source'] not in {'swde', 'kleister-nda', 'kleister-charity', 'vrdu'}:
        return row, []
    result = deepcopy(row)
    audit = []
    for task in result['tasks']:
        gold = result['gold'][task['id']]
        remove = {}
        for key in task['fields']:
            answers = [r[key] for r in gold['records']]
            if any(a['status'] != 'present' for a in answers):
                remove[key] = 'source_absence_not_verified_locally'
            elif any(not explicit_local_role(row['text'], key, s, row['source'])
                     for a in answers for s in a['items']):
                remove[key] = 'unsupported_local_field_role'
        for key, reason in remove.items():
            task['fields'].pop(key)
            task.get('cardinalities', {}).pop(key, None)
            for record in gold['records']:
                record.pop(key)
            audit.append({'id': row['id'], 'split': row['split'], 'source': row['source'],
                          'task': task['id'], 'field': key, 'reason': reason})
        if row['source'] == 'kleister-nda' and 'jurisdiction' in task['fields']:
            task['fields']['jurisdiction'] = 'Jurisdiction explicitly named for governing law or courts, not place of incorporation or geographic coverage'
    result['tasks'] = [t for t in result['tasks'] if t['fields'] and (t['kind'] != 'records' or t['anchor'] in t['fields'])]
    result['gold'] = {t['id']: result['gold'][t['id']] for t in result['tasks']}
    if not result['tasks']:
        return None, audit
    if row['source'] == 'kleister-charity':
        positive = {k for gold in result['gold'].values() for record in gold['records']
                    for k, answer in record.items() if answer['items']}
        if not positive & FINANCIAL_FIELDS and result['family'] != 'metadata':
            result['family'] = 'metadata'
            audit.append({'id': row['id'], 'split': row['split'], 'source': row['source'],
                          'reason': 'organization_metadata_not_financial_metrics'})
    return result, audit
