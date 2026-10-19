"""Literal public-source field contracts; no inferred labels or normalized values."""
VRDU_FIELDS = {
    'property': 'Name of the broadcasting station issuing this advertising order or invoice',
    'tv_address': 'Printed postal address of the issuing broadcasting station',
    'advertiser': 'Advertiser named for this advertising order, not the station or agency',
    'agency': 'Advertising agency placing the order, not the advertiser or station owner',
    'product': 'Advertised product or campaign named in the order header',
    'contract_num': 'Advertising order or contract identifier in the header',
    'gross_amount': 'Gross total for the whole advertising order, not a line amount, net total or commission',
    'flight_from': 'Start date of the overall advertising campaign in the header, not a line broadcast date',
    'flight_to': 'End date of the overall advertising campaign in the header, not a line broadcast date',
    'program_desc': 'Program or broadcast slot description for this line item',
    'program_start_date': 'Start or broadcast date of this line item',
    'program_end_date': 'End date of this line item',
    'sub_amount': 'Printed charge for this line item',
    'channel': 'Broadcasting channel identifier for this line item',
    'registrant_name': 'Name of the person or organization registered as foreign agent, not a notary',
    'registration_num': 'Foreign agent registration identifier, excluding a preceding # sign',
    'foreign_principle_name': 'Name of the foreign principal represented by the registered agent',
    'signer_name': 'Name of the registrant or authorized representative signing the declaration, not the notary',
    'signer_title': 'Job title of the registrant representative signing the declaration, not the notary title',
    'file_date': 'Printed date of the registrant declaration or its notarization, not receipt stamp or notary commission expiry',
}
CORD_FIELDS = {
    'menu_nm': 'Entire printed item name or description on this receipt line',
    'menu_price': 'Entire printed line-item price, including any printed currency or field label',
    'menu_cnt': 'Printed quantity of this receipt item',
    'menu_unitprice': 'Entire printed unit price of this receipt item, including any printed label',
    'total_total_price': 'Entire printed receipt grand-total field, INCLUDING its printed label (such as TOTAL or TL) if present',
    'total_cashprice': 'Entire printed cash-tendered field, INCLUDING its printed label (such as CASH) if present',
    'total_changeprice': 'Entire printed change-returned field, INCLUDING its printed label if present',
}


def description(source, key):
    mapping = CORD_FIELDS if source == 'cord' else VRDU_FIELDS
    if key in mapping:
        return mapping[key]
    label = key.replace('_', ' ')
    return f'Entire printed {label} field, including its printed label if present' if source == 'cord' else label


def complete_token_boundary(text, start, end):
    """Reject annotations cutting through a printed identifier/date token."""
    def token(char):
        return char.isalnum() or char in '/-'
    return not ((start and token(text[start-1]) and token(text[start]))
                or (end < len(text) and token(text[end-1]) and token(text[end])))


def split_lookup(registry, group):
    from pathlib import Path

    from .coverage_assembly import sha
    from .jsonio import loads

    active = registry['active_splits'][group]
    root = Path(registry['path'])
    paths = [active] + sorted(p for p in registry['files']
                             if p.startswith(group+'/few_shot-splits/') and p != active)
    lookup = {}
    # Previously used source splits override the larger split: no heldout row
    # ever becomes training data and no prior training row becomes new evidence.
    for name in paths:
        if sha(root/name) != registry['files'][name]:
            raise ValueError('Pinned split changed')
        data = loads((root/name).read_text())
        for key, split in [('train', 'train'), ('valid', 'dev'), ('test', 'test')]:
            for ident in data.get(key, []):
                lookup[ident] = split
    return lookup
