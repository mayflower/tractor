"""Build a real-text extraction-only corpus; no model inference or synthetic padding."""

import argparse
import hashlib
import multiprocessing
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from .corpus_expansion import example_key, source_rows
from .episodes import Episode, text_key
from .generalist_data import read_episodes, validate_native
from .jsonio import dumps, loads
from .source_import import envelope, qa

FEW_TYPES = ['O', 'art', 'building', 'event', 'location', 'organization', 'other', 'person', 'product']
FEW_DESCRIPTIONS = {
    'art': 'named work of art, music, film or writing',
    'building': 'named building or facility', 'event': 'named event',
    'location': 'named geographical location', 'organization': 'named organization',
    'other': 'other named entity, including awards, diseases, laws and languages',
    'person': 'named person', 'product': 'named product or vehicle',
}
PUBLIC_SOURCES = {'germanquad', 'squad_v2', 'multispan', 'docred', 'few_nerd', 'adversarial_qa'}


def few_nerd(row):
    """Original token sequence, space-joined deterministically; no guessed detokenization.

    The dataset has flat fine-type runs, not BIO tags. Use fine types to preserve
    boundaries between adjacent mentions mapped into the same coarse type.
    Original document IDs are unavailable, so this adapter is training-only.
    """
    tokens, tags, fine = row['tokens'], row['ner_tags'], row['fine_ner_tags']
    if not tokens or len(tokens) != len(tags) or len(tags) != len(fine):
        raise ValueError('Token/tag length mismatch')
    if any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError('Empty/non-string token')
    if any(not isinstance(t, int) or not 0 <= t < len(FEW_TYPES) for t in tags):
        raise ValueError('Unknown coarse tag')
    if any(not isinstance(t, int) or not 0 <= t <= 66 for t in fine):
        raise ValueError('Unknown fine tag')
    text, positions = '', []
    for token in tokens:
        if text:
            text += ' '
        positions.append(len(text))
        text += token
    gold = {k: {'status': 'absent', 'items': []} for k in FEW_DESCRIPTIONS}
    index = 0
    while index < len(tokens):
        last = index + 1
        while last < len(tokens) and fine[last] == fine[index] and tags[last] == tags[index]:
            last += 1
        if bool(tags[index]) != bool(fine[index]):
            raise ValueError('Inconsistent outside tags')
        if tags[index]:
            key = FEW_TYPES[tags[index]]
            start, end = positions[index], positions[last-1] + len(tokens[last-1])
            gold[key]['status'] = 'present'
            gold[key]['items'].append({'start': start, 'end': end, 'text': text[start:end]})
        index = last
    episode = envelope('few_nerd', row['id'], 'few_nerd:sentence:' + text_key(text), text,
                       'en', 'named_entities', 'source_tokens_space_joined', 'human',
                       [{'id': 'entities', 'kind': 'entities', 'description': 'Named entities',
                         'fields': FEW_DESCRIPTIONS}],
                       {'entities': {'complete': True, 'records': [gold]}})
    return episode.model_copy(update={'split': 'train'})


def require_extraction(episode):
    from .extraction_scope import in_scope

    if not in_scope(episode.model_dump()):
        raise ValueError('Out-of-scope task cannot count as public extraction')
    if episode.source not in PUBLIC_SOURCES or episode.text_origin != 'real' or episode.annotation != 'human':
        raise ValueError('Only allowlisted real public human-annotated extraction sources count')
    for task in episode.tasks:
        gold = episode.gold[task.id]
        if task.kind == 'classification' or not gold.complete:
            raise ValueError('Classification/incomplete annotation does not count')
        for record in gold.records:
            for answer in record.values():
                if answer.label is not None or answer.status not in {'present', 'absent'}:
                    raise ValueError('Only spans or explicit absence count')
                if answer.status == 'present' and not (answer.items or answer.answer_alternatives):
                    raise ValueError('Present extraction target requires evidence')


def word_shingles(text):
    words = re.findall(r'\w+', text.casefold())
    return {' '.join(words[i:i+5]) for i in range(len(words)-4)}


class ProtectedIndex:
    """All protected shingles indexed; candidate minimum signatures find containment.

    Unlike the older audit, do not discard common buckets or short five-word texts.
    This remains lexical screening, not proof against semantic paraphrase leakage.
    """
    def __init__(self):
        self.groups, self.texts = set(), set()
        self.index, self.shingles = defaultdict(set), []

    def add(self, episode):
        self.groups.update((episode.source_group, episode.derivation_group))
        key = text_key(episode.text)
        if key in self.texts:
            return
        self.texts.add(key)
        sh = word_shingles(episode.text)
        ident = len(self.shingles)
        self.shingles.append(sh)
        for item in sh:
            self.index[item].add(ident)

    def overlaps(self, episode):
        if {episode.source_group, episode.derivation_group} & self.groups:
            return 'protected_source_group'
        if text_key(episode.text) in self.texts:
            return 'protected_normalized_text'
        sh = word_shingles(episode.text)
        signature = sorted(sh, key=lambda x: hashlib.sha256(x.encode()).digest())[:8]
        candidates = set()
        for item in signature:
            candidates.update(self.index.get(item, ()))
        for ident in candidates:
            other = self.shingles[ident]
            if len(sh & other) / min(len(sh), len(other)) >= .9:
                return 'protected_lexical_containment'
        return None


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(config, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    index = ProtectedIndex()
    inputs = {}
    for path in config['protected_episodes']:
        inputs[path] = checksum(path)
        for e in read_episodes(path):
            index.add(e)
    blocked = set(loads(Path(config['blocked_groups']).read_text()))
    counts, seen, conflicts, heldout = Counter(), {}, set(), Counter()
    rejection = root.joinpath('filter-rejections.jsonl').open('x')

    def all_rows():
        for path in config['existing_train']:
            inputs[path] = checksum(path)
            for e in read_episodes(path):
                if e.source in PUBLIC_SOURCES:
                    yield e
        for spec in config['new_sources']:
            inputs[spec['path']] = checksum(spec['path'])
            if inputs[spec['path']] != spec['sha256']:
                raise ValueError('Pinned source checksum mismatch')
            for row in source_rows(spec['path']):
                try:
                    e = few_nerd(row) if spec['source'] == 'few_nerd' else qa(row, spec['source'])
                except (ValueError, KeyError, IndexError) as exc:
                    counts[spec['source'] + ':adapter_rejected:' + str(exc)[:100]] += 1
                    continue
                if e.split != 'train':
                    heldout[e.source + ':' + e.split] += 1
                    continue
                yield e

    with rejection, root.joinpath('candidates.episodes.jsonl').open('x') as stream:
        for e in all_rows():
            require_extraction(e)
            if e.split != 'train' or e.original_split != 'train':
                raise ValueError('Training-only input required')
            counts[e.source + ':read'] += 1
            reason = 'quarantined_source_group' if e.source_group in blocked else index.overlaps(e)
            identity = hashlib.sha256(dumps({'text': ' '.join(e.text.split()).casefold(),
                'tasks': [t.model_dump() for t in e.tasks]}, sort_keys=True).encode()).hexdigest()
            example = example_key(e)
            if identity in seen:
                reason = 'duplicate_task' if seen[identity] == example else 'conflicting_annotation'
                if reason == 'conflicting_annotation':
                    conflicts.add(identity)
            if reason:
                counts[e.source + ':' + reason] += 1
                rejection.write(dumps({'id': e.id, 'reason': reason, 'identity': identity}) + '\n')
                continue
            seen[identity] = example
            stream.write(dumps(e.model_dump()) + '\n')
            counts[e.source + ':candidate'] += 1
            if sum(v for k, v in counts.items() if k.endswith(':candidate')) % 10000 == 0:
                print('FILTER', dict(counts), flush=True)
    # If duplicated tasks disagree, neither version is eligible.
    (root / 'conflicting-identities.json').write_text(dumps(sorted(conflicts)))
    (root / 'preparation.json').write_text(dumps({'config': config, 'inputs': inputs,
        'counts': dict(counts), 'heldout_new_source_rows_not_trained': dict(heldout),
        'protected_texts': len(index.texts), 'protected_groups': len(index.groups),
        'conflicting_identities': len(conflicts),
        'screen': 'all protected word-5-shingles; eight candidate minimum signatures; containment >=0.9, plus exact normalized text and source/derivation groups'}, indent=2))


_EXTRACTOR = None


def init_validator(model):
    global _EXTRACTOR
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch

    from .model import Extractor
    torch.set_num_threads(1)
    _EXTRACTOR = Extractor(model, 'cpu')


def check_row(line):
    episode = Episode.model_validate(loads(line))
    from .types import TractorError
    try:
        native, tokens = validate_native(episode, _EXTRACTOR)
        return line, dumps(native), tokens, None
    except (ValueError, RuntimeError, TractorError) as exc:
        return line, None, 0, type(exc).__name__ + ':' + str(exc)[:160]


def validate(output, model, workers):
    root = Path(output)
    counts, sources, families, languages = Counter(), Counter(), Counter(), Counter()
    texts, groups, ids = set(), set(), set()
    conflicts = set(loads((root / 'conflicting-identities.json').read_text()))

    def candidates():
        with (root / 'candidates.episodes.jsonl').open() as stream:
            for line in stream:
                if conflicts:
                    e = loads(line)
                    identity = hashlib.sha256(dumps({'text': ' '.join(e['text'].split()).casefold(),
                        'tasks': e['tasks']}, sort_keys=True).encode()).hexdigest()
                    if identity in conflicts:
                        counts['conflicting_annotation_quarantine'] += 1
                        continue
                yield line

    with (root / 'train.episodes.jsonl').open('x') as ef, (root / 'train.native.jsonl').open('x') as nf, \
            (root / 'native-rejections.jsonl').open('x') as rf, \
            multiprocessing.get_context('spawn').Pool(workers, initializer=init_validator, initargs=(model,)) as pool:
        for line, native, tokens, error in pool.imap(check_row, candidates(), chunksize=32):
            e = loads(line)
            if error:
                counts[e['source'] + ':rejected:' + error] += 1
                rf.write(dumps({'id': e['id'], 'reason': error}) + '\n')
                continue
            if e['id'] in ids:
                raise ValueError('Duplicate accepted id')
            ids.add(e['id'])
            ef.write(line)
            nf.write(native + '\n')
            sources[e['source']] += 1
            families[e['family']] += 1
            languages[e['language']] += 1
            texts.add(text_key(e['text']))
            groups.add(e['source_group'])
            counts['accepted'] += 1
            counts['joint_tokens'] += tokens
            if counts['accepted'] % 10000 == 0:
                print('VALIDATED', counts['accepted'], dict(sources), flush=True)
    report = {'status': 'complete' if counts['accepted'] >= 200000 else 'below_required_minimum',
        'train_examples': counts['accepted'], 'unique_texts': len(texts), 'source_groups': len(groups),
        'sources': dict(sources), 'families': dict(families), 'languages': dict(languages),
        'origins': {'real/human': counts['accepted']}, 'counts': dict(counts),
        'model_sha256': checksum(Path(model) / 'model.safetensors'),
        'files': {name: checksum(root / name) for name in ['train.episodes.jsonl', 'train.native.jsonl']},
        'qualification': 'All rows structurally and native-collator validated at 512 joint tokens. No inference, training, calibration or final evaluation. Human labels are source annotations, not a guarantee of correctness. Few-NERD preserves source tokens with explicit space joining; original document IDs unavailable, new NER source kept train-only. Lexical screening is not semantic independence proof.'}
    (root / 'manifest.json').write_text(dumps(report, indent=2))
    print(dumps(report, indent=2), flush=True)
    if counts['accepted'] < 200000:
        raise ValueError('Required minimum of 200000 validated training examples not reached')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'validate'])
    parser.add_argument('--config', default='configs/public-extraction.json')
    parser.add_argument('--output', default='data/public-extraction-200k')
    parser.add_argument('--model', default='models/tractor')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(loads(Path(args.config).read_text()), args.output)
    else:
        validate(args.output, args.model, args.workers)


if __name__ == '__main__':
    main()
