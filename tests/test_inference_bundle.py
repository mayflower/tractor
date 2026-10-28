"""Relocatable inference bundles: explicit selection, full hashes, strict loading."""
import hashlib
import shutil

import pytest

from tractor import biqwen, inference_bundle
from tractor.inference_bundle import (
    MANIFEST,
    export_bundle,
    load_inference_bundle,
    verify_inference_bundle,
)
from tractor.jsonio import dumps, loads


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    base = tmp_path / "train-host/base"
    files = {"bootstrap.py": b"# runtime", "model/extractor_config.json": b"{}",
             "model/backbone/model.safetensors": b"base weights"}
    for name, data in files.items():
        (base / name).parent.mkdir(parents=True, exist_ok=True)
        (base / name).write_bytes(data)
    lock = tmp_path / "lock.json"
    lock.write_text(dumps({"repo": "test/biqwen", "revision": "rev1",
                           "files": {k: digest(v) for k, v in files.items()}}))
    monkeypatch.setattr(biqwen, "LOCK", lock)
    run = tmp_path / "train-host/run/step-00000010"
    saved = {"adapter/adapter_model.safetensors": b"adapter", "adapter/adapter_config.json": b"{}",
             "heads.safetensors": b"heads", "training-state.pt": b"optimizer state"}
    for name, data in saved.items():
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_bytes(data)
    metadata = {"format": "biqwen-full-epochs-fast-v4", "base_model": str(base), "base_revision": "rev1",
                "base_weights": {}, "architecture": "biqwen-joint-record-v2", "schema_version": "biqwen-types-v2",
                "attention": {"mode": "qwen3.5-hybrid-bidirectional-v1"}, "step": 10, "temperatures": {"source": 1.0},
                "config": {"attention_mode": "qwen3.5-hybrid-bidirectional-v1", "output_dir": "/private/path",
                           "runtime": {"gdn_backend": "fla"}},
                "files": {k: digest(v) for k, v in saved.items()}}
    (run / "checkpoint.json").write_text(dumps(metadata))
    return run


def test_export_is_relocatable_complete_and_excludes_training_state(checkpoint, tmp_path):
    output = tmp_path / "export/bundle"
    summary = export_bundle(checkpoint, output, "abc123")
    manifest = loads((output / MANIFEST).read_text())
    assert "checkpoint/training-state.pt" not in manifest["files"]
    assert not (output / "checkpoint/training-state.pt").exists()
    assert manifest["checkpoint"]["manifest_sha256"] == inference_bundle.sha(checkpoint / "checkpoint.json")
    text = (output / MANIFEST).read_text()
    assert str(tmp_path) not in text and "/private/path" not in text
    assert manifest["contract"]["gdn_backend"] == "fla" and manifest["tractor_commit"] == "abc123"
    assert summary["files"] == len(manifest["files"])
    # Relocate, and remove the original training folder entirely.
    moved = tmp_path / "elsewhere/bundle"
    shutil.move(output, moved)
    shutil.rmtree(tmp_path / "train-host")
    assert verify_inference_bundle(moved)["base_revision"] == "rev1"
    with pytest.raises(FileExistsError):
        export_bundle(checkpoint, moved, "abc123")


@pytest.mark.parametrize("damage", ["changed", "extra", "missing_head", "base_revision", "absolute"])
def test_broken_bundles_fail_verification(checkpoint, tmp_path, damage):
    output = tmp_path / "bundle"
    export_bundle(checkpoint, output, "abc123")
    manifest = loads((output / MANIFEST).read_text())
    if damage == "changed":
        (output / "checkpoint/heads.safetensors").write_bytes(b"other")
    elif damage == "extra":
        (output / "checkpoint/notes.txt").write_text("x")
    elif damage == "missing_head":
        (output / "checkpoint/heads.safetensors").unlink()
        del manifest["files"]["checkpoint/heads.safetensors"]
    elif damage == "base_revision":
        manifest["base"]["revision"] = "rev2"
    else:
        manifest["files"]["/etc/passwd"] = manifest["files"].pop("checkpoint/heads.safetensors")
    (output / MANIFEST).write_text(dumps(manifest))
    with pytest.raises(ValueError):
        verify_inference_bundle(output)


def test_loader_uses_only_bundle_artifacts(checkpoint, tmp_path, monkeypatch):
    from tractor import biqwen_training

    output = tmp_path / "bundle"
    export_bundle(checkpoint, output, "abc123")
    shutil.rmtree(tmp_path / "train-host")
    calls = []

    class Base:
        model_revision, weights_hashes = "rev1", {}

        def __init__(self, path, device, precision):
            calls.append(("base", path))

    monkeypatch.setattr(biqwen, "BiQwenExtractor", Base)
    monkeypatch.setattr(biqwen_training, "apply_checkpoint",
                        lambda backend, contract, adapter, heads, device: calls.append((adapter, heads)))
    backend = load_inference_bundle(output)
    assert calls == [("base", output.resolve() / "base"),
                     (output.resolve() / "checkpoint/adapter", output.resolve() / "checkpoint/heads.safetensors")]
    assert backend.model_revision.startswith("bundle:")
    assert set(backend.weights_hashes) == {"base/model/backbone/model.safetensors",
                                           "checkpoint/adapter/adapter_model.safetensors",
                                           "checkpoint/heads.safetensors"}


def test_incomplete_biqwen_layouts_never_fall_back_to_gliner(tmp_path, monkeypatch):
    from tractor import model
    from tractor.backends import load_extractor

    monkeypatch.setattr(model, "Extractor", lambda *args: "gliner")
    assert load_extractor(tmp_path) == "gliner"
    (tmp_path / "heads.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="Incomplete BiQwen"):
        load_extractor(tmp_path)
