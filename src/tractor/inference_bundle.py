"""Relocatable, fully hashed BiQwen inference bundles.

A training checkpoint names its merged base by absolute path and lists
training-state.pt among its files. This export instead copies an explicit
selection (pinned base, adapter, heads) under one new manifest with its own
hashes, and identifies the original checkpoint manifest by its unchanged hash.
The training checkpoint and base bundle are never modified. Verification is
pure Python; loading needs the BiQwen extras and never touches the network.
"""

import hashlib
import shutil
from pathlib import Path

from .jsonio import dumps, loads

FORMAT = "tractor-biqwen-inference-v1"
MANIFEST = "inference-bundle.json"
CHECKPOINT_FILES = ("heads.safetensors",)
CHECKPOINT_DIRS = ("adapter",)
# Markers of a BiQwen layout. Such a directory is never loaded as GLiNER.
BIQWEN_MARKERS = (MANIFEST, "checkpoint.json", "bootstrap.py", "tractor_qwen", "adapter",
                  "heads.safetensors", "base", "model/extractor_config.json")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lock():
    from .biqwen import LOCK

    return loads(LOCK.read_text())


def export_bundle(checkpoint, output, commit):
    """Copy the explicit inference selection of one verified checkpoint."""
    from .biqwen import DECODER_VERSION, verify_bundle
    from .biqwen_training import inference_contract

    checkpoint, output = Path(checkpoint).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    metadata = loads((checkpoint / "checkpoint.json").read_text())
    for name, digest in metadata["files"].items():
        if sha(checkpoint / name) != digest:
            raise ValueError(f"Checkpoint artifact changed: {name}")
    base = Path(metadata["base_model"])
    pinned = verify_bundle(base)
    if metadata["base_revision"] != pinned["revision"]:
        raise ValueError("Checkpoint base differs from the pinned BiQwen revision")
    selected = [name for name in metadata["files"]
                if name in CHECKPOINT_FILES or name.split("/", 1)[0] in CHECKPOINT_DIRS]
    if "heads.safetensors" not in selected or not any(n.startswith("adapter/") for n in selected):
        raise ValueError("Checkpoint lacks adapter or heads")
    temporary = output.with_name(output.name + ".writing")
    temporary.mkdir(parents=True)
    for name in pinned["files"]:
        target = temporary / "base" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / name, target)
    for name in selected:
        target = temporary / "checkpoint" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint / name, target)
    contract = inference_contract(metadata)
    manifest = {
        "format": FORMAT,
        "tractor_commit": commit,
        "backend": DECODER_VERSION,
        "checkpoint": {"format": metadata["format"], "step": metadata.get("step"),
                       "manifest_sha256": sha(checkpoint / "checkpoint.json")},
        "contract": contract,
        "base": {"repo": pinned["repo"], "revision": pinned["revision"], "files": pinned["files"]},
        "files": {p.relative_to(temporary).as_posix(): sha(p)
                  for p in sorted(temporary.rglob("*")) if p.is_file()},
        # Only demonstrated capabilities; uncertainty heads need their own artifacts.
        "capabilities": ["source_spans", "typed_extraction", "typed_lists"],
        "uncertainty": None,
    }
    (temporary / MANIFEST).write_text(dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_inference_bundle(temporary)
    temporary.rename(output)
    return {"bundle": str(output), "manifest_sha256": sha(output / MANIFEST),
            "files": len(manifest["files"])}


def verify_inference_bundle(path):
    """Every file hashed, nothing unlisted, base equal to the package lock."""
    path = Path(path)
    manifest = loads((path / MANIFEST).read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported inference bundle format")
    pinned = lock()
    if manifest["base"]["revision"] != pinned["revision"] or manifest["base"]["files"] != pinned["files"]:
        raise ValueError("Bundle base differs from the pinned merged BiQwen base")
    listed = manifest["files"]
    for name in pinned["files"]:
        if listed.get(f"base/{name}") != pinned["files"][name]:
            raise ValueError(f"Bundle base file is missing or differs: {name}")
    if "checkpoint/heads.safetensors" not in listed or not any(n.startswith("checkpoint/adapter/") for n in listed):
        raise ValueError("Bundle lacks adapter or heads")
    present = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} - {MANIFEST}
    if present != set(listed):
        raise ValueError(f"Bundle files differ from manifest: {sorted(present ^ set(listed))[:5]}")
    for name, digest in listed.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(path / name) != digest:
            raise ValueError(f"Bundle artifact changed: {name}")
    uncertainty = manifest.get("uncertainty")
    if uncertainty is not None:
        from .uncertainty_artifacts import verify_binding

        verify_binding(uncertainty, manifest)
    return {"bundle": str(path), "format": FORMAT, "manifest_sha256": sha(path / MANIFEST),
            "base_revision": manifest["base"]["revision"], "files": len(listed),
            "capabilities": manifest["capabilities"]}


def load_inference_bundle(path, device="cpu", precision="fp32"):
    from .biqwen import BiQwenExtractor
    from .biqwen_training import apply_checkpoint

    path = Path(path).resolve()
    summary = verify_inference_bundle(path)
    manifest = loads((path / MANIFEST).read_text())
    backend = BiQwenExtractor(path / "base", device, precision)
    apply_checkpoint(backend, manifest["contract"], path / "checkpoint/adapter",
                     path / "checkpoint/heads.safetensors", device)
    backend.model_revision = "bundle:" + summary["manifest_sha256"]
    backend.weights_hashes = {k: v for k, v in manifest["files"].items() if k.endswith(".safetensors")}
    return backend
