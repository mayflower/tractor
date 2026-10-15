"""Create allowlisted source/model snapshots without Git history or local experiments."""

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'dist'
SOURCE_FILES = (
    'README.md', 'AGENTS.md', 'CONTRIBUTING.md', 'Dockerfile', 'justfile',
    'pyproject.toml', 'uv.lock', 'flake.nix', 'flake.lock', '.envrc',
    '.gitignore', '.dockerignore',
)
SOURCE_DIRS = ('src', 'tests', 'scripts', 'examples', 'docs', 'configs', 'release', 'benchmarks')
MODEL_FILES = (
    'model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
    'thresholds.json', 'normalization.json', 'source-lock.json', 'README.md', 'metrics.json',
    'tractor.svg', 'comparison.svg',
)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def files_under(directory):
    return [p for p in directory.rglob('*') if p.is_file()
            and not any(part.startswith('.') or part == '__pycache__'
                        for part in p.relative_to(directory).parts)
            and p.suffix not in ('.pyc', '.pyo')]


def pack(name, base, files, mode):
    path = OUTPUT / name
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    manifest = {}
    with tarfile.open(path, mode) as tar:
        for source in sorted(files):
            if source.is_symlink():
                raise ValueError(f'Symlinks are not release artifacts: {source}')
            relative = source.relative_to(base).as_posix()
            manifest[relative] = digest(source)
            info = tar.gettarinfo(str(source), arcname='tractor/'+relative)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ''
            info.mode = 0o644
            with source.open('rb') as stream:
                tar.addfile(info, stream)
    return {'archive': name, 'sha256': digest(path), 'files': manifest}


def main():
    OUTPUT.mkdir(exist_ok=True)
    targets = json.loads((ROOT/'release/targets.json').read_text())
    if not (targets['github_private'] and targets['huggingface_private']):
        raise ValueError('Both repository targets must be private')
    model = ROOT / targets['local_bundle']
    if digest(model/'model.safetensors') != targets['weights_sha256']:
        raise ValueError('Release model weights differ from the selected artifact')
    source_files = [ROOT/name for name in SOURCE_FILES]
    source_files += [p for name in SOURCE_DIRS for p in files_under(ROOT/name)]
    if (ROOT/'LICENSE').exists():
        source_files.append(ROOT/'LICENSE')
    model_files = [model/name for name in MODEL_FILES] + files_under(model/'encoder_config')
    if not (model/'encoder_config/config.json').is_file():
        raise ValueError('Missing embedded encoder configuration')
    manifests = [pack('tractor-source.tar.gz', ROOT, source_files, 'w:gz'),
                 pack('tractor-model.tar', model, model_files, 'w')]
    (OUTPUT/'manifest.json').write_text(json.dumps(manifests, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in m.items() if k != 'files'} for m in manifests], indent=2))


if __name__ == '__main__':
    main()
