"""Fetch or verify the pinned public files used by document_sources; never run dataset code."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    specs = json.loads(Path('configs/document-sources.json').read_text())
    for spec in specs:
        if 'path' not in spec:
            print(spec['source'], spec['status'])
            continue
        root = Path(spec['path'])
        for name, digest in spec['files'].items():
            path = root / name
            if not path.exists() and not args.verify_only:
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f'https://raw.githubusercontent.com/{spec["repo"]}/{spec["revision"]}/{name}'
                temporary = path.with_suffix(path.suffix+'.download')
                with urllib.request.urlopen(url, timeout=120) as response:
                    temporary.write_bytes(response.read())
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                    raise ValueError('Downloaded source checksum mismatch')
                temporary.replace(path)
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Missing or changed pinned source: {path}')
        print(spec['source'], 'verified', len(spec['files']), 'files')


if __name__ == '__main__':
    main()
