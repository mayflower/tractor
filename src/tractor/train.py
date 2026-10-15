"""Native full fine-tuning with weight/gradient evidence and explicit provenance."""

import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

from .data import read_rows
from .jsonio import dumps
from .model import Extractor
from .training_adapter import configure_processor


def train(model, data, config_path, device="cuda"):
    import torch
    from gliner2.training import ExtractorTrainer, TrainingConfig

    config_data = json.loads(Path(config_path).read_text())
    if device == "cpu" and (config_data["bf16"] or config_data["fp16"]):
        raise ValueError("CPU training requires an explicit fp32 configuration")
    config = TrainingConfig(**config_data)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run directory already populated; choose a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    size = sum(p.stat().st_size for p in Path(model).glob("*.safetensors"))
    required = size * (min(config.num_epochs, config.save_total_limit) + 3) + 2 * 1024**3
    if shutil.disk_usage(output).free < required:
        raise ValueError(
            f"Insufficient disk space: need at least {required / 1024**3:.1f} GiB free"
        )
    x = Extractor(model, device)
    configure_processor(x.model.processor)
    # Watch representative parameters from each trainable subsystem.
    selected = {}
    for name, param in x.model.named_parameters():
        if param.requires_grad and param.numel() > 100:
            group = name.split(".")[0]
            if group not in selected:
                selected[group] = (
                    name,
                    param,
                    param.detach().flatten()[:1024].float().cpu().clone(),
                )
    gradients = {key: False for key in selected}
    handles = []
    for key, (_, param, _) in selected.items():

        def capture(gradient, key=key):
            if torch.isfinite(gradient).all() and gradient.abs().sum().item() > 0:
                gradients[key] = True

        handles.append(param.register_hook(capture))
    activation_dtypes = set()
    loss_components = {}
    for module in x.model.encoder.modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, value: activation_dtypes.add(str(value.dtype))
                )
            )
            break

    def observe_loss(_module, _inputs, value):
        if len(loss_components) < 32 and value.losses:
            for key, loss in value.losses.items():
                if torch.is_tensor(loss) and loss.numel() == 1:
                    loss_components[key] = float(loss.detach())

    handles.append(x.model.register_forward_hook(observe_loss))
    native_train = read_rows(Path(data) / "train.native.jsonl")
    native_dev = read_rows(Path(data) / "dev.native.jsonl")
    trainer = ExtractorTrainer(x.model, config)
    if str(trainer.device) != device and not (
        device == "cuda" and str(trainer.device).startswith("cuda")
    ):
        raise ValueError("Native trainer selected an unexpected device")
    started = time.monotonic()
    result = trainer.train(train_data=native_train, eval_data=native_dev)
    changes = {
        key: {
            "parameter": name,
            "changed": not torch.equal(before, param.detach().flatten()[:1024].float().cpu()),
            "nonzero_finite_gradient": gradients[key],
        }
        for key, (name, param, before) in selected.items()
    }
    for handle in handles:
        handle.remove()
    report = {
        "result": result,
        "elapsed_seconds": time.monotonic() - started,
        "parameters": changes,
        "device": str(trainer.device),
        "dtype": str(next(trainer.model.parameters()).dtype),
        "encoder_linear_activation_dtypes": sorted(activation_dtypes),
        "observed_loss_components": loss_components,
        "config": config_data,
        "base_model": str(Path(model).resolve()),
        "dataset_sha256": hashlib.sha256(Path(data, "train.native.jsonl").read_bytes()).hexdigest(),
        "versions": {
            k: importlib.metadata.version(k) for k in ["torch", "transformers", "gliner2"]
        },
        "resume": "Weights-only restart; optimizer/scheduler/RNG resume unsupported by pinned trainer.",
    }
    Path(output, "training-evidence.json").write_text(dumps(report, indent=2))
    for key in ("encoder", "boundary_head", "record_decoder"):
        if not changes[key]["changed"] or not changes[key]["nonzero_finite_gradient"]:
            raise RuntimeError(f"{key} learning not demonstrated")
    return report
