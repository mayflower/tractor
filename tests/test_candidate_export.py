import hashlib
import json

from tractor.evaluate import export


def test_export_excludes_training_state_and_private_paths(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "final"
    (checkpoint / "encoder_config").mkdir(parents=True)
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "encoder_config/config.json"]:
        (checkpoint / name).write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"test weights")
    (checkpoint / "optimizer.pt").write_bytes(b"private training state")
    (run / "training-evidence.json").write_text(json.dumps({"result": {"total_steps": 1},
        "required_learning_groups": ["encoder"],
        "parameters": {"encoder": {"changed": True, "nonzero_finite_gradient": True}},
        "base_model": "/private/base", "initial_weights_sha256": "a" * 64}))
    selection = run / "selection.json"
    selection.write_text(json.dumps({"model": str(checkpoint), "thresholds": {"text": .5, "quantity": .5, "anchor": .5},
        "predictions": "/private/predictions.jsonl", "split": "dev", "metrics": {"field_f1": .8}}))
    output = tmp_path / "candidate"
    export(run, selection, output)
    assert not (output / "optimizer.pt").exists()
    assert not (output / "training-evidence.json").exists()
    assert "/private" not in "".join(p.read_text() for p in output.glob("*.json"))
    lock = json.loads((output / "source-lock.json").read_text())
    assert lock["weights_sha256"] == hashlib.sha256(b"test weights").hexdigest()
    assert lock["parent_weights_sha256"] == "a" * 64
