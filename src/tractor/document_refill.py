"""Append candidates for unmet quotas without changing authored data or its recipe."""
import argparse
import fcntl
import math
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .coverage_assembly import sha
from .document_build import plan_item
from .jsonio import dumps, loads


def refill(root):
    # Caller must hold campaign.lock. No requests are sent by this operation.
    root = Path(root)
    status = loads((root / 'campaign-status.json').read_text())
    if status['status'] != 'planned_candidates_exhausted':
        raise ValueError('Refill requires confirmed exhausted candidates')
    manifest = loads((root / 'manifest.json').read_text())
    recipe = loads((root / 'recipe.json').read_text())
    audit = loads((root / 'scale-audit.json').read_text())
    previous_recipe = sha(root / 'recipe.json')
    if not audit['passed'] or audit['recipe_sha256'] != previous_recipe:
        raise ValueError('Existing authoring audit is not valid')
    if sha(root / 'plan.jsonl') != recipe['plan_sha256'] or recipe['plan_sha256'] != manifest['files']['plan.jsonl']:
        raise ValueError('Existing plan changed')
    for key, filename in [('code_sha256', 'document_build.py'), ('facts_code_sha256', 'document_facts.py'), ('scenario_code_sha256', 'document_scenarios.py')]:
        if recipe[key] != sha(Path(__file__).with_name(filename)):
            raise ValueError('Refill cannot change authoring code')
    for name, prompt in recipe['prompts'].items():
        if Path(f'configs/document-{name}.txt').read_text() != prompt:
            raise ValueError('Refill cannot change authoring prompts')
    totals, maxima, schema_hashes = Counter(), {}, {}
    for line in (root / 'plan.jsonl').open():
        item = loads(line)
        totals[':'.join(item[k] for k in ['split', 'domain', 'shape', 'language'])] += 1
        pair = item['split'], item['domain']
        maxima[pair] = max(maxima.get(pair, -1), int(item['id'].rsplit(':', 1)[1]))
        schema_hashes[item['domain']] = item['schema_sha256']
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
    history = root / 'refills' / stamp
    history.mkdir(parents=True, exist_ok=False)
    for name in ['manifest.json', 'recipe.json', 'scale-audit.json', 'campaign-status.json']:
        shutil.copy2(root / name, history / name)
    (history / 'plan.jsonl').hardlink_to(root / 'plan.jsonl')
    pending = root / 'plan.refill.tmp'
    shutil.copyfile(root / 'plan.jsonl', pending)
    added = Counter()
    with pending.open('a') as stream:
        for key, cell in status['cells'].items():
            missing = cell['target'] - cell['accepted']
            if missing <= 0:
                continue
            split, domain, shape, language = key.split(':')
            spec = loads(Path('schemas/extraction', domain + '.json').read_text())
            import hashlib
            if hashlib.sha256(dumps(spec, sort_keys=True).encode()).hexdigest() != schema_hashes[domain]:
                raise ValueError('Refill cannot change field schemas')
            # Conservative empirical yield; only the quota cells still short
            # receive new candidates. Rejects never count toward accepted quotas.
            rate = max(.02, cell['accepted'] / max(1, totals[key]))
            count = max(32, math.ceil(missing / rate * 1.25))
            pair = split, domain
            while added[key] < count:
                maxima[pair] += 1
                index = maxima[pair]
                if ('de' if (index // 10 + index) % 2 == 0 else 'en') != language:
                    continue
                item = plan_item(spec, split, index, manifest['config']['seed'], shape)
                stream.write(dumps(item) + '\n')
                added[key] += 1
                manifest['counts'][f'{split}:{domain}'] += 1
    pending.replace(root / 'plan.jsonl')
    new_hash = sha(root / 'plan.jsonl')
    manifest['files']['plan.jsonl'] = new_hash
    recipe['plan_sha256'] = new_hash
    (root / 'manifest.json').write_text(dumps(manifest, indent=2))
    (root / 'recipe.json').write_text(dumps(recipe, indent=2))
    amendment = {'previous_recipe_sha256': previous_recipe, 'recipe_sha256': sha(root / 'recipe.json'),
                 'added_candidates': dict(added), 'total_added': sum(added.values()),
                 'qualification': 'Append-only candidate extension for unmet cells. Existing plan prefix, cached authoring, '
                 'prompts, schemas, code, accepted outputs and quality criteria unchanged. Original audit continues to apply '
                 'to the identical authoring method; every new case still requires both reviewers and native validation.'}
    audit['recipe_sha256'] = amendment['recipe_sha256']
    audit.setdefault('plan_extensions', []).append(amendment)
    (root / 'scale-audit.json').write_text(dumps(audit, indent=2))
    (history / 'extension.json').write_text(dumps(amendment, indent=2))
    print(dumps({'refill': str(history), 'added_candidates': sum(added.values())}), flush=True)
    return amendment


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    with Path(args.root, 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refill(args.root)
