"""Remove table-position supervision while preserving literal header/text extraction."""
import argparse
import re
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .episodes import Episode, native_episode
from .jsonio import dumps, loads
from .source_policy import excluded_source


def position_blocks(text):
    """Recognize numbered item blocks with field lists, not numbered prose.

    This is a corpus scope check only. It never groups runtime extractions or
    consults model predictions/answers. Single-object forms remain supported.
    """
    heading = re.compile(r'^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:position|pos\.?|item|line(?:\s+item)?|artikel)'
                         r'\s*(?:no\.?\s*|nr\.?\s*|#\s*)?\d+\b[^\n]*$', re.I)
    field = re.compile(r'^\s*(?:[-*]\s+)?(?:\*\*)?[^:;.!?\n]{1,60}:\s*\S')
    item_label = re.compile(r'^\s*(?:[-*]\s+|\d+[.)]\s*)?(?:artikel|ware|warenbeschreibung|produkt|'
                            r'item|goods|product|description of goods)\s*:\s*\S', re.I)
    blocks, labelled, current = [], [], None
    inline_positions = 0
    for line in text.splitlines():
        if heading.fullmatch(line):
            inline_positions += bool(re.search(r'\d+\s*:\s*\S', line))
            current = []
            blocks.append(current)
        elif current is not None:
            current.append(line)
    current = None
    for line in text.splitlines():
        if item_label.match(line):
            current = []
            labelled.append(current)
        elif current is not None:
            current.append(line)
    return (inline_positions >= 2
            or sum(sum(bool(field.match(line)) for line in block) >= 2 for block in blocks) >= 2
            or sum(sum(bool(field.match(line)) for line in block) >= 1 for block in labelled) >= 2)


def table_record_reason(row):
    if not any(t['kind'] == 'records' for t in row.get('tasks', [])):
        return None
    if row['source'] in {'vrdu', 'docile'}:
        return 'public_layout_position_task'
    # Conservative structural quarantine, never a prediction-quality filter.
    # Incidental tables can quarantine prose records too; retain header tasks.
    lines = row.get('text', '').splitlines()
    if any(line.count('\t') >= 2 or line.count('|') >= 2 for line in lines):
        return 'delimited_table_with_record_task'
    if sum(len(re.split(r' {2,}', line.strip())) >= 3 for line in lines) >= 2:
        return 'aligned_columns_with_record_task'
    if position_blocks(row.get('text', '')):
        return 'labelled_position_blocks_with_record_task'
    return None


def restrict(row):
    if excluded_source(row):
        return None, [{'id': row['id'], 'source': row['source'], 'split': row['split'],
                       'task': t['id'], 'reason': 'excluded_source'} for t in row['tasks']]
    reason = table_record_reason(row)
    if reason is None:
        return row, []
    removed = [t['id'] for t in row['tasks'] if t['kind'] == 'records']
    tasks = [t for t in row['tasks'] if t['id'] not in removed]
    audit = [{'id': row['id'], 'source': row['source'], 'split': row['split'],
              'task': ident, 'reason': reason} for ident in removed]
    if not tasks:
        return None, audit
    result = dict(row, tasks=tasks, gold={t['id']: row['gold'][t['id']] for t in tasks},
                  artifact='single_object')
    if 'processing' in row:
        result['processing'] = {t['id']: row['processing'][t['id']] for t in tasks
                                if t['id'] in row['processing']}
    return result, audit


def prepare(source, output, roles=False):
    source, output = Path(source), Path(output)
    manifest = loads((source/'manifest.json').read_text())
    if not manifest.get('training_ready'):
        raise ValueError('Input must have passed its existing qualification')
    output.mkdir(parents=True, exist_ok=False)
    counts, removed, groups = Counter(), Counter(), set()
    with (output/'excluded-tasks.jsonl').open('w') as audit:
        for split in ('train', 'dev', 'calibration', 'test'):
            en, nn = f'{split}.episodes.jsonl', f'{split}.native.jsonl'
            for name in (en, nn):
                if sha(source/name) != manifest['files'][name]:
                    raise ValueError('Source hash changed')
            with (source/en).open() as ef, (source/nn).open() as nf, \
                    (output/en).open('w') as eo, (output/nn).open('w') as no:
                for eline, nline in zip(ef, nf, strict=True):
                    old = loads(eline)
                    if old['split'] != split or old['text'] != loads(nline)['input']:
                        raise ValueError('Unpaired input')
                    row, rejected = restrict(old)
                    if roles and row is not None:
                        from .document_role_scope import restrict as restrict_roles
                        row, role_audit = restrict_roles(row)
                        rejected.extend(role_audit)
                    for item in rejected:
                        audit.write(dumps(item)+'\n')
                        removed[f'{split}:{item["source"]}:{item["reason"]}'] += 1
                    if row is None:
                        continue
                    if rejected:
                        episode = Episode.model_validate(row)
                        native, skipped = native_episode(episode)
                        if native is None or skipped:
                            raise ValueError('Remaining tasks lost supervision')
                        eo.write(dumps(episode.model_dump())+'\n')
                        no.write(dumps(native)+'\n')
                    else:
                        eo.write(eline)
                        no.write(nline)
                    counts[split] += 1
                    if split == 'train' and row['text_origin'] == 'real':
                        groups.add(row['source_group'])
    report = dict(manifest, status='text_scope_filtered', counts=dict(counts),
                  accepted_by_split=dict(counts), source_manifest_sha256=sha(source/'manifest.json'),
                  removed_tasks=dict(removed), scope='Explicit text extraction; no table/layout reconstruction',
                  local_role_audit=roles,
                  role_policy_sha256=sha(Path(__file__).with_name('document_role_scope.py')) if roles else None,
                  scope_limit='Structural quarantine is conservative, not semantic proof. Existing prose record review retained; ambiguous cases need review.',
                  files={p.name: sha(p) for p in output.glob('*.jsonl')})
    if 'unique_real_train' in manifest:
        report['unique_real_train'] = len(groups)
    # Source-group summaries from the parent are stale after task removal.
    report.pop('source_groups', None)
    (output/'manifest.json').write_text(dumps(report, indent=2)+'\n')
    print(dumps({'output': str(output), 'counts': dict(counts), 'removed': dict(removed),
                 'unique_real_train': len(groups)}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--roles', action='store_true')
    prepare(**vars(parser.parse_args()))
