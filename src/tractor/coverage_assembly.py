"""Assemble validated public and gated synthetic corpora without changing source splits."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from .episodes import Episode, text_key
from .jsonio import dumps, loads


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assemble(base, synthetic, output, paraphrases=None, role_evaluation=None, classification=None):
    from .extraction_scope import in_scope

    base, synthetic, output = map(Path, (base, synthetic, output))
    bm = loads((base / 'manifest.json').read_text())
    sm = loads((synthetic / 'manifest.json').read_text())
    if bm['status'] != 'complete' or sm['status'] != 'validated' or sm['scale_status'] != 'passed_matched_dev_pilot':
        raise ValueError('Assembly requires a complete base and a validated, gated scale corpus')
    sources = {}
    for split in ('train', 'dev', 'calibration', 'test'):
        base_dir = base if split in ('train', 'dev') else base / 'supplement'
        sources[split] = [(base_dir / f'{split}.episodes.jsonl', base_dir / f'{split}.native.jsonl'),
                          (synthetic / f'{split}.episodes.jsonl', synthetic / f'{split}.native.jsonl')]
    expected = {str(base / name): value for name, value in bm['files'].items()}
    expected.update({str(synthetic / name): value for name, value in sm['files'].items()})
    if classification:
        classification_root = Path(classification)
        classification_manifest = loads((classification_root / 'manifest.json').read_text())
        if classification_manifest['status'] != 'validated':
            raise ValueError('Classification supplement has not passed native validation')
        for split in sources:
            sources[split].append((classification_root / f'{split}.episodes.jsonl',
                                   classification_root / f'{split}.native.jsonl'))
        expected.update({str(classification_root / name): value
                         for name, value in classification_manifest['files'].items()})
    if role_evaluation:
        role_root = Path(role_evaluation)
        role_manifest = loads((role_root / 'manifest.json').read_text())
        if role_manifest['status'] != 'validated':
            raise ValueError('Additional role evaluation has not passed native validation')
        for split in ('dev', 'calibration', 'test'):
            sources[split].append((role_root / f'{split}.episodes.jsonl', role_root / f'{split}.native.jsonl'))
        expected.update({str(role_root / name): value for name, value in role_manifest['files'].items()})
    if paraphrases:
        root = Path(paraphrases)
        review = loads((root / 'agent-review.json').read_text())
        if sha(root / 'accepted.episodes.jsonl') != review['input_sha256']:
            raise ValueError('Paraphrase review no longer matches accepted candidates')
        approved = {r['id'] for r in review['decisions'] if r['accepted']}
        actual = {loads(line)['id'] for line in (root / 'curated.episodes.jsonl').read_text().splitlines()}
        if approved != actual:
            raise ValueError('Curated paraphrases do not match review decisions')
        pair = (root / 'curated.episodes.jsonl', root / 'curated.native.jsonl')
        sources['train'].append(pair)
        expected.update({str(p): sha(p) for p in pair})
    for pairs in sources.values():
        for pair in pairs:
            for p in pair:
                if str(p) not in expected or sha(p) != expected[str(p)]:
                    raise ValueError(f'Unverified source file: {p}')
    output.mkdir(parents=True, exist_ok=False)
    ids, groups, texts = set(), {}, {}
    counts, origins, families, languages = Counter(), Counter(), Counter(), Counter()
    for split, pairs in sources.items():
        with (output / f'{split}.episodes.jsonl').open('w') as ef, (output / f'{split}.native.jsonl').open('w') as nf:
            for ep, native in pairs:
                with ep.open() as es, native.open() as ns:
                    for line, native_line in zip(es, ns, strict=True):
                        episode = Episode.model_validate(loads(line))
                        if episode.split != split or episode.id in ids:
                            raise ValueError('Invalid split or duplicate episode ID during assembly')
                        if loads(native_line)['input'] != episode.text:
                            raise ValueError('Native/episode rows do not refer to the same text')
                        if not in_scope(episode.model_dump()):
                            continue
                        ids.add(episode.id)
                        for group in (episode.source_group, episode.derivation_group):
                            if group in groups and groups[group] != split:
                                raise ValueError('Source group crosses assembled splits')
                            groups[group] = split
                        key = text_key(episode.text)
                        if key in texts and texts[key] != split:
                            raise ValueError('Normalized text crosses assembled splits')
                        texts[key] = split
                        ef.write(line)
                        nf.write(native_line)
                        counts[split] += 1
                        origins[f'{split}:{episode.text_origin}/{episode.label_origin}'] += 1
                        families[f'{split}:{episode.family}'] += 1
                        languages[f'{split}:{episode.language}'] += 1
    report = {'status': 'complete', 'counts': dict(counts), 'origins': dict(origins),
              'families': dict(families), 'languages': dict(languages),
              'unique_texts': len(texts), 'source_groups': len(groups),
              'source_file_hashes': expected,
              'qualification': 'Training concatenates validated sources; shuffle during training. '
              'Calibration/test combine the structured-source and synthetic supplements plus optional classification; '
              'they do not replace the separately protected human-public final evaluation corpus. '
              'No full-corpus training, final calibration/test inference, upload or release promotion.',
              'files': {p.name: sha(p) for p in output.glob('*.jsonl')}}
    (output / 'manifest.json').write_text(dumps(report, indent=2))
    print(dumps({'status': report['status'], 'counts': report['counts'], 'origins': report['origins']}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='data/generalist-extraction-integrated')
    parser.add_argument('--synthetic', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--paraphrases')
    parser.add_argument('--role-evaluation')
    parser.add_argument('--classification')
    args = parser.parse_args()
    assemble(args.base, args.synthetic, args.output, args.paraphrases, args.role_evaluation, args.classification)
