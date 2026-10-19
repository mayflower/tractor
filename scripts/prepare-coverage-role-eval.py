"""Prepare additional, unscored held-out cases for expanded scalar-role contrasts."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tractor.episodes import audit_episodes
from tractor.jsonio import dumps
from tractor.public_extraction import check_row, init_validator
from tractor.synthetic_coverage import make_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    languages = ['de'] * 9 + ['en'] * 9 + ['mixed'] * 2
    for split in ('dev', 'calibration', 'test'):
        for pair in range(500):
            for variant in (0, 1):
                episode, _ = make_episode(split, 20000 + 2 * pair, 'role_contrasts',
                                          languages[pair % len(languages)], variant=variant)
                rows.append(episode)
    audit = audit_episodes(rows)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_validator, initargs=('models/tractor',)) as pool:
        validated = list(pool.map(check_row, (dumps(e.model_dump()) for e in rows), chunksize=32))
    errors = [error for _, _, _, error in validated if error]
    if errors:
        (root / 'rejections.json').write_text(dumps(errors, indent=2))
        raise ValueError('New role cases failed native validation')
    for split in ('dev', 'calibration', 'test'):
        with (root / f'{split}.episodes.jsonl').open('w') as ef, (root / f'{split}.native.jsonl').open('w') as nf:
            for episode, (line, native, _, _) in zip(rows, validated, strict=True):
                if episode.split == split:
                    ef.write(line + '\n')
                    nf.write(native + '\n')
    import hashlib
    (root / 'manifest.json').write_text(dumps({'status': 'validated', 'audit': audit,
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.jsonl')},
        'qualification': 'Additional held-out scalar-role cases, not used by the original pilot gate. '
        'No model predictions; screen against expanded train before combining.'}, indent=2))


if __name__ == '__main__':
    main()
