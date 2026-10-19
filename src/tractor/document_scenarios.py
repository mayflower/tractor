"""Authorship briefs, not document templates: vary document use and presentation."""

PRESENTATIONS = [
    'A compact export from a business application, with short headings and explicitly described facts',
    'A document excerpt beginning with the operative paragraph, followed by factual prose',
    'A text-only document with restrained section numbering and a plain closing line',
    'A printed document transcribed into text, with a heading, plain header and body',
    'A professional document using short factual paragraphs and a compact closing block',
    'An attachment excerpt with a descriptive title and a concise introductory sentence',
    'A document with a compact header and natural sentences describing each entry',
    'A document with the substantive content first and reference/contact information at the end',
    'A document laid out in short sections, using prose for context and explicit sentences for repeated entries',
    'An excerpt with ordinary abbreviations customary in this industry and clear record boundaries',
    'A document in simple language for a non-specialist recipient, with concise descriptive headings',
    'An internal copy of a completed business document, with brief notes and substantive content',
    'A document with a letterhead, short body and conventional signature/footer block',
    'A document whose repeated entries are compact individual blocks rather than a grid',
    'A document with a short factual preamble and an explicitly headed annex or schedule',
    'A plain-text document suitable for an accessible document viewer, with linear reading order',
    'A document excerpt with section captions and locally conventional dates and amounts',
    'A document containing one brief operational note, the main information and a closing reference',
    'A completed document with a concise header and a short explanatory paragraph after the details',
    'A compact business document with a descriptive heading, main content and recipient-appropriate closing',
]

CONTEXTS = {
 'invoices': ['industrial spare parts', 'laboratory consumables', 'office furniture', 'commercial cleaning', 'electrical installation',
              'printing services', 'machine maintenance', 'wholesale catering supplies', 'building materials', 'IT equipment',
              'protective workwear', 'horticultural supplies', 'packaging materials', 'vehicle workshop supplies', 'event equipment hire',
              'freight forwarding', 'retail shop fittings', 'measuring instruments', 'hospitality supplies', 'water treatment equipment'],
 'contracts': ['commercial premises', 'warehouse lease', 'office maintenance services', 'equipment hire', 'cleaning services',
               'facilities management', 'laboratory premises', 'retail shop lease', 'commercial kitchen lease', 'parking facilities',
               'industrial workshop lease', 'HVAC maintenance', 'security system maintenance', 'grounds maintenance', 'printing equipment hire',
               'commercial storage', 'building inspection services', 'coworking premises', 'machine servicing', 'commercial laundry services'],
 'banking': ['vehicle purchase financing', 'home improvement financing', 'appliance purchase financing', 'education expenses', 'debt consolidation',
             'equipment financing for a sole trader', 'household relocation expenses', 'personal loan renewal', 'account review supporting a loan',
             'new loan application', 'joint household borrowing', 'short-term personal financing', 'credit application update', 'salary account review',
             'loan application supporting documents', 'branch loan application summary', 'credit file update', 'self-employed income statement',
             'loan application with recent transactions', 'existing customer borrowing request'],
 'communication': ['delivery rescheduling', 'replacement parts request', 'installation appointment', 'order quantity correction', 'service visit booking',
                   'shipping address update', 'callback request about an order', 'backorder follow-up', 'equipment collection appointment',
                   'purchase order clarification', 'request for additional supplies', 'delivery slot confirmation', 'maintenance parts request',
                   'warehouse pickup coordination', 'repair appointment confirmation', 'amended order request', 'subscription equipment return',
                   'site access coordination', 'urgent spare-parts availability enquiry', 'follow-up on a quotation'],
}


def brief(domain, index, split):
    # Index cycles independently from 20-case shape/language cadence.
    variant = (index // 20 + index * 7) % 20
    return {'family': f'{domain}:{split}:presentation-{variant:02d}',
            'presentation': PRESENTATIONS[variant],
            'business_context': CONTEXTS.get(domain, [domain])[variant % len(CONTEXTS.get(domain, [domain]))],
            'instruction': 'Adapt the presentation to the actual genre. Write an original document, not a description of a layout. Do not add filler to satisfy a word target.'}
