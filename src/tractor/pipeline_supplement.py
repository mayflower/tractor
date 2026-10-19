"""Small reviewed training-only supplement: role contrasts and explicit prose records."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .coverage_assembly import sha
from .document_build import plan_item, render
from .document_critic import review
from .episodes import Episode, audit_episodes
from .jsonio import dumps, loads


def prepare(root, focus=False):
    root.mkdir(parents=True, exist_ok=False)
    if focus:
        rows = []
        for domain, count in [('research', 160), ('careers', 48), ('contacts', 48)]:
            spec = loads(Path('schemas/extraction', domain+'.json').read_text())
            for i in range(count):
                item = plan_item(spec, 'train', 1000000+i, 2026101903,
                                 shape='repeated_records', record_count=2+i % 3)
                item['authorship_brief'] = str(item['authorship_brief']) + (
                    ' Write realistic professional prose, with locally explicit fields for each separately named record. '
                    'No tables, aligned columns, key-value lists, implicit relations or inferred facts. '
                    'Vary sentence order and professional vocabulary. Do not reuse example names or a fixed text template. '
                    'Research: explicitly name each measured property, its value with unit, and its condition for each sample. '
                    'Careers: explicitly state the job title at each employer and the stated employment dates. '
                    'Contacts: explicitly connect each named contact with their own role and email in prose.')
                item['document_genre'] = {
                    'research': 'Laboratory measurement narrative with separate sample observations',
                    'careers': 'Professional biography with explicitly described employment history',
                    'contacts': 'Business correspondence introducing named contacts and their responsibilities',
                }[domain]
                rows.append(item)
        (root/'plan.jsonl').write_text(''.join(dumps(r)+'\n' for r in rows))
        (root/'manifest.json').write_text(dumps({'status': 'planned',
            'files': {'plan.jsonl': sha(root/'plan.jsonl')}, 'training_only': True,
            'candidate_limit': len(rows), 'focus': 'rare prose record fields'}, indent=2))
        return
    fields = {
        'charity_name': ('Full legal name of the reporting charity', 'Oakbridge Community Trust'),
        'charity_number': ('Registered charity number of the reporting charity, not company number', '1234567'),
        'company_number': ('Company registration number of the reporting charity, not charity number', '08765432'),
        'address__street_line': ('Registered office street, not visitor or contact address', 'Spencer Road'),
        'address__post_town': ('Registered office town, not visitor or contact town', 'London'),
        'address__postcode': ('Registered office postcode, not visitor or contact postcode', 'EC1A 9BU'),
        'visitor_address': ('Separately stated visitor or service address', '18 Harbour Lane, Bristol BS1 4DJ'),
    }
    spec = {'domain': 'metadata', 'document': 'Registered charity details in a UK annual-report excerpt',
            'header': {k: {'description': d, 'examples': [v, v]} for k, (d, v) in fields.items()}}
    rows = []
    for i in range(64):
        item = plan_item(spec, 'train', 800000+i, 2026101902, shape='single_object')
        item['authorship_brief'] = str(item['authorship_brief']) + ' Fictional UK charity report. Include clearly distinguished charity and company numbers and registered and visitor offices. Vary street forms including named streets without house numbers. Use UK addresses in both languages. No inferred facts, tables or field-list templates.'
        item['document_genre'] = 'UK charity annual-report prose excerpt'
        rows.append(item)
    domains = list(loads(Path('configs/document-build.json').read_text())['domains'])
    for i in range(192):
        domain = domains[i % len(domains)]
        spec = loads(Path('schemas/extraction', domain+'.json').read_text())
        item = plan_item(spec, 'train', 900000+i, 2026101902, shape='repeated_records', record_count=2+i % 3)
        item['authorship_brief'] = str(item['authorship_brief']) + ' Write coherent professional prose with explicitly described separate records and local field roles. Distinguish each record through its named anchor; do not use tables, column layouts, field-list templates, implicit links or calculations. Vary order and sentence structure naturally.'
        rows.append(item)
    (root/'plan.jsonl').write_text(''.join(dumps(r)+'\n' for r in rows))
    (root/'manifest.json').write_text(dumps({'status': 'planned', 'files': {'plan.jsonl': sha(root/'plan.jsonl')},
                                           'training_only': True, 'candidate_limit': 256}, indent=2))


def run(root, env='~/.env', workers=8, focus=False):
    root = Path(root)
    if not root.exists():
        prepare(root, focus=focus)
    manifest = loads((root/'manifest.json').read_text())
    render(root, limit=manifest['candidate_limit'], env=env, workers=workers)
    rows = [loads(p.read_text()) for p in sorted((root/'reviewed').glob('*.json'))]
    candidates = [r for r in rows if r['accepted_by_teacher']]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reviews = list(pool.map(lambda r: review(r, root, env), candidates))
    accepted = []
    for row, decision in zip(candidates, reviews, strict=True):
        if not decision['passed']:
            continue
        episode = row['episode']
        if row['plan']['document_genre'] == 'UK charity annual-report prose excerpt':
            episode['source'] = 'tractor_authored_gb_offices'
        accepted.append(Episode.model_validate(episode))
    audit = audit_episodes(accepted)
    if any(e.split != 'train' for e in accepted):
        raise ValueError('Supplement must be training only')
    (root/'accepted.episodes.jsonl').write_text(''.join(dumps(e.model_dump())+'\n' for e in accepted))
    report = {'candidates': len(rows), 'teacher_accepted': len(candidates), 'independently_accepted': len(accepted),
              'audit': audit, 'sha256': hashlib.sha256((root/'accepted.episodes.jsonl').read_bytes()).hexdigest(),
              'qualification': 'Two-provider realism/semantic review; native capacity and text-scope checks still required before training. Not evaluation evidence.'}
    (root/'accepted.json').write_text(dumps(report, indent=2))
    print(dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--env', default='~/.env')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--focus', action='store_true')
    run(**vars(p.parse_args()))
