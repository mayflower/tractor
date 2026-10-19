"""Marker-preserving Teacher candidates; semantic review remains Silver evidence."""

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path

from .episodes import Episode
from .generalist_data import read_episodes, validate_native
from .jsonio import dumps, loads

STYLES = ('informal email with greeting and sign-off', 'concise professional prose',
          'internal handover note', 'short dialogue between colleagues', 'compact bullet-point memo')
MARKER = re.compile(r'<v(\d+)>(.*?)</v\1>', re.DOTALL)


def mark(episode):
    if episode.split != 'train' or episode.text_origin != 'synthetic':
        raise ValueError('Paraphrases may only derive from synthetic training episodes')
    spans = {}
    for gold in episode.gold.values():
        for record in gold.records:
            for answer in record.values():
                if answer.status not in {'present', 'absent'} or answer.answer_alternatives:
                    raise ValueError('Only unambiguous complete targets are supported')
                for span in answer.items:
                    spans[(span.start, span.end)] = span.text
    result, cursor, mapping = '', 0, []
    for index, ((start, end), value) in enumerate(sorted(spans.items())):
        if start < cursor:
            raise ValueError('Overlapping targets cannot be safely marked')
        result += episode.text[cursor:start] + f'<v{index}>{value}</v{index}>'
        mapping.append({'start': start, 'end': end, 'text': value})
        cursor = end
    return result + episode.text[cursor:], mapping


def reconstruct(original, response):
    _, mapping = mark(original)
    if response.get('id') != original.id:
        raise ValueError('Teacher returned the wrong episode ID')
    text = response['marked_text']
    found, plain, cursor, targets = set(), '', 0, {}
    for match in MARKER.finditer(text):
        index = int(match[1])
        if index in found or index >= len(mapping) or match[2] != mapping[index]['text']:
            raise ValueError('Teacher changed or duplicated a target')
        found.add(index)
        plain += text[cursor:match.start()]
        start = len(plain)
        plain += match[2]
        targets[(mapping[index]['start'], mapping[index]['end'])] = {
            'start': start, 'end': len(plain), 'text': match[2]}
        cursor = match.end()
    plain += text[cursor:]
    if len(found) != len(mapping) or re.search(r'</?v\d+', plain):
        raise ValueError('Missing or malformed target markers')
    for value in {span['text'] for span in mapping}:
        pattern = r'(?<!\w)' + re.escape(value) + r'(?!\w)'
        if len(re.findall(pattern, plain)) != len(re.findall(pattern, original.text)):
            raise ValueError('Teacher added or removed an unmarked target occurrence')
    data = original.model_dump()
    for gold in data['gold'].values():
        for record in gold['records']:
            for answer in record.values():
                answer['items'] = [targets[(span['start'], span['end'])] for span in answer['items']]
    data.update(id=original.id + '-paraphrase', text=plain, annotation='silver', label_origin='teacher',
                source='tractor_coverage_paraphrases', artifact='teacher_paraphrase')
    return Episode.model_validate(data)


def prepare(input_path, output, per_family=10):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    selected, counts = [], Counter()
    for episode in read_episodes(input_path):
        if counts[episode.family] >= per_family:
            continue
        counts[episode.family] += 1
        selected.append(episode)
    (root / 'originals.episodes.jsonl').write_text(''.join(dumps(e.model_dump()) + '\n' for e in selected))
    with (root / 'render-input.jsonl').open('w') as out:
        for index, episode in enumerate(selected):
            text, _ = mark(episode)
            out.write(dumps({'id': episode.id, 'split': 'train', 'text_origin': 'synthetic',
                             'style': STYLES[index % len(STYLES)], 'language': episode.language,
                             'tasks': [t.model_dump() for t in episode.tasks], 'marked_text': text}) + '\n')
    (root / 'provenance.json').write_text(dumps({
        'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        'counts': dict(counts), 'status': 'render_candidates_only'}, indent=2))


def request_ids(path, prompt_path):
    prompt = Path(prompt_path).read_text()
    result = {}
    for line in Path(path).read_text().splitlines():
        item = loads(line)
        key = hashlib.sha256(dumps({"model": "claude-haiku-5-5", "prompt": prompt, "input": item}).encode()).hexdigest()
        result[key] = item['id']
    return result


def candidates(output):
    root = Path(output)
    originals = {e.id: e for e in read_episodes(root / 'originals.episodes.jsonl')}
    rejected = []
    expected = request_ids(root / 'render-input.jsonl', 'configs/coverage-render.txt')
    with (root / 'candidates.episodes.jsonl').open('x') as ef, (root / 'verify-input.jsonl').open('x') as vf:
        for path in sorted((root / 'renders').glob('*.json')):
            cached = loads(path.read_text())
            result = cached['result']
            try:
                expected_id = expected[cached['input_sha256']]
                if result.get('id') != expected_id:
                    raise ValueError('Teacher response ID does not match its request')
                original = originals[expected_id]
                episode = reconstruct(original, result)
            except (KeyError, ValueError) as exc:
                rejected.append({'file': path.name, 'error': str(exc)})
                continue
            ef.write(dumps(episode.model_dump()) + '\n')
            vf.write(dumps({'id': original.id, 'split': 'train', 'text_origin': 'synthetic',
                            'original': original.model_dump(), 'revised': episode.model_dump()}) + '\n')
    (root / 'render-rejections.json').write_text(dumps(rejected, indent=2))


def accept(output, model):
    import torch

    from .model import Extractor
    from .types import TractorError

    torch.set_num_threads(1)
    root = Path(output)
    reviews = {}
    expected = request_ids(root / 'verify-input.jsonl', 'configs/coverage-verify.txt')
    for path in (root / 'reviews').glob('*.json'):
        cached = loads(path.read_text())
        result = cached['result']
        expected_id = expected[cached['input_sha256']]
        if result.get('id') == expected_id:
            reviews[expected_id] = result
    extractor = Extractor(model, 'cpu')
    counts = Counter()
    with (root / 'accepted.episodes.jsonl').open('x') as ef, (root / 'accepted.native.jsonl').open('x') as nf:
        for episode in read_episodes(root / 'candidates.episodes.jsonl'):
            review = reviews.get(episode.id.removesuffix('-paraphrase'), {})
            if review.get('semantic_pass') is not True or review.get('issues') != []:
                counts['semantic_rejected_or_missing'] += 1
                continue
            try:
                native, _ = validate_native(episode, extractor)
            except (ValueError, RuntimeError, TractorError):
                counts['native_rejected'] += 1
                continue
            ef.write(dumps(episode.model_dump()) + '\n')
            nf.write(dumps(native) + '\n')
            counts['accepted_silver'] += 1
    (root / 'validation.json').write_text(dumps({'counts': dict(counts),
        'qualification': 'Literal targets preserved; second Teacher pass checked semantic roles. '
        'Same-model review is not independent human gold. Training-only supplement; not part of frozen pilot arms.'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'candidates', 'accept'])
    parser.add_argument('--input', default='data/synthetic-coverage-pilot/train.episodes.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='models/tractor')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.input, args.output)
    elif args.phase == 'candidates':
        candidates(args.output)
    else:
        accept(args.output, args.model)
