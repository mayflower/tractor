"""Download the pinned BiQwen runtime and weights explicitly; no automatic downloads at serve time."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from tractor.biqwen import LOCK, verify_bundle
from tractor.jsonio import loads

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='models/biqwen-extract-decide-4b')
args = parser.parse_args()
lock = loads(LOCK.read_text())
path = snapshot_download(lock['repo'], revision=lock['revision'], local_dir=args.output,
                         allow_patterns=[*lock['files'], 'README.md', 'tractor_qwen/LICENSE.gliner2'])
verify_bundle(Path(path))
print(f'Verified {lock["repo"]}@{lock["revision"]} in {path}')
