"""Reproducible near-duplicate audit with heldout-first source-group quarantine.

64-permutation MinHash LSH proposes pairs; exact five-token shingle Jaccard
confirms them. Both literal and gold-masked text are checked. This is a detector,
not a proof that semantically paraphrased templates are independent.
"""
import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from .coverage_assembly import sha
from .jsonio import dumps, loads


def words(text):
    return re.findall(r'\w+|[^\w\s]', text.casefold())


def masked(row):
    spans = sorted({(s['start'], s['end']) for gold in row['gold'].values()
                    for record in gold['records'] for answer in record.values() for s in answer['items']})
    text, end = '', 0
    for start, stop in spans:
        if start < end:
            continue
        text += row['text'][end:start] + ' VALUE '
        end = stop
    return text + row['text'][end:]


def shingles(text):
    tokens = words(text)
    return {' '.join(tokens[i:i+5]) for i in range(max(1, len(tokens)-4))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


class Index:
    def __init__(self):
        import numpy as np
        self.np = np
        rng = np.random.default_rng(20261024)
        self.a = rng.integers(1, 2**63, 64, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**63, 64, dtype=np.uint64)
        self.bands = defaultdict(list)
        self.rows = []

    def check(self, row, threshold=.85, cross_only=False, *, add=True):
        np = self.np
        pair = (shingles(row['text']), shingles(masked(row)))
        keys, candidates = [], set()
        for mode, values in enumerate(pair):
            hashes = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little')
                               for s in values], dtype=np.uint64)
            signature = (hashes[:, None] * self.a + self.b).min(axis=0)
            for band in range(16):
                key = (mode, band, signature[band*4:band*4+4].tobytes())
                keys.append(key)
                candidates.update(self.bands[key])
        matches = []
        for ident in sorted(candidates):
            prior, prior_pair = self.rows[ident]
            if prior['source_group'] == row['source_group']:
                continue
            if cross_only and prior['split'] == row['split']:
                continue
            scores = [jaccard(a, b) for a, b in zip(pair, prior_pair, strict=True)]
            # Masked boilerplate is useful to detect synthetic template reuse.
            # On public receipts it erases almost the entire document, so it
            # must not establish leakage without matching literal evidence.
            masked_allowed = row.get('text_origin') != 'real' and prior.get('text_origin') != 'real'
            if scores[0] >= threshold or (masked_allowed and scores[1] >= threshold):
                matches.append({'id': row['id'], 'prior_id': prior['id'], 'source_group': row['source_group'],
                                'prior_source_group': prior['source_group'], 'split': row['split'],
                                'prior_split': prior['split'], 'literal_jaccard': scores[0], 'masked_jaccard': scores[1]})
        if not add:
            return matches
        ident = len(self.rows)
        self.rows.append(({**{k: row[k] for k in ['id', 'source_group', 'split']}, 'text_origin': row.get('text_origin')}, pair))
        for key in keys:
            self.bands[key].append(ident)
        return matches


def audit(source, output, threshold=.85, cross_only=False):
    root = Path(source)
    index = Index()
    findings, inputs = [], {}
    for split in ('test', 'calibration', 'dev', 'train'):
        path = root / f'{split}.episodes.jsonl'
        inputs[path.name] = sha(path)
        for line in path.open():
            findings.extend(index.check(loads(line), threshold, cross_only))
    report = {'inputs': inputs, 'threshold': threshold, 'cross_only': cross_only, 'checked': len(index.rows), 'findings': findings,
              'cross_split_groups_to_quarantine': sorted({r['source_group'] for r in findings if r['split'] != r['prior_split']}),
              'qualification': 'Probabilistic candidate retrieval, exact shingle confirmation. Gold-masked text tests template reuse. '
              'Same-split similarity is reported separately; cross-split groups require quarantine/refill before qualification.'}
    with Path(output).open('x') as stream:
        stream.write(dumps(report, indent=2))
    print(dumps({'checked': report['checked'], 'findings': len(findings),
                 'cross_split_groups': len(report['cross_split_groups_to_quarantine'])}), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threshold', type=float, default=.85)
    p.add_argument('--cross-only', action='store_true')
    audit(**vars(p.parse_args()))
