"""Native full fine-tuning with weight/gradient evidence and explicit provenance."""

import fcntl
import hashlib
import importlib.metadata
import json
import math
import resource
import shutil
import time
from pathlib import Path

from .data import read_rows
from .jsonio import dumps
from .model import load_native
from .training_adapter import configure_processor


def parameter_hash(parameter):
    values = parameter.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def train(model, data, config_path, device="cuda"):
    manifest_path = Path(data, "manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("training_ready") is False:
            raise ValueError("Dataset is explicitly not training-ready; resolve its audit findings first")
    lock_path = Path("runs/.training.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(model, data, config_path, device)


def _train(model, data, config_path, device):
    import torch
    from gliner2.training import ExtractorTrainer, TrainingConfig

    config_data = json.loads(Path(config_path).read_text())
    evidence_groups = config_data.pop("tractor_required_groups", ["encoder", "boundary_head", "record_decoder"])
    max_seconds = config_data.pop("tractor_max_seconds", 7200)
    generalist_dev = config_data.pop("tractor_generalist_dev", None)
    extraction_dev = config_data.pop("tractor_extraction_dev", None)
    dev_thresholds = config_data.pop("tractor_dev_thresholds", {})
    if dev_thresholds and not extraction_dev:
        raise ValueError('Dev thresholds require production pipeline evaluation')
    if generalist_dev and extraction_dev:
        raise ValueError('Choose one Dev evaluator')
    quality_policy = config_data.pop("tractor_quality_policy", "configs/generalist-quality.json")
    if device == "cpu" and (config_data["bf16"] or config_data["fp16"]):
        raise ValueError("CPU training requires an explicit fp32 configuration")
    config = TrainingConfig(**config_data)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA training is unavailable; CPU fallback is disabled")
        if config.bf16 and not torch.cuda.is_bf16_supported():
            raise ValueError("Requested BF16 is unsupported on the selected GPU")
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run directory already populated; choose a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(file.name.encode() + b"\0" + file.read_bytes())
    with Path(model, "model.safetensors").open("rb") as stream:
        initial_weights_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    size = sum(p.stat().st_size for p in Path(model).glob("*.safetensors"))
    required = size * (min(config.num_epochs, config.save_total_limit) + 3) + 2 * 1024**3
    if shutil.disk_usage(output).free < required:
        raise ValueError(
            f"Insufficient disk space: need at least {required / 1024**3:.1f} GiB free"
        )
    native_model = load_native(model, device)
    configure_processor(native_model.processor)
    # Watch representative parameters from each trainable subsystem.
    selected = {}
    for name, param in native_model.named_parameters():
        if param.requires_grad and param.numel() > 100:
            group = name.split(".")[0]
            if group not in selected:
                selected[group] = (
                    name,
                    param,
                    parameter_hash(param),
                )
    gradients = {key: False for key in selected}
    handles = []
    workload = {"training_samples": 0, "training_tokens": 0, "padded_tokens": 0}

    def count_work(module, inputs):
        if module.training:
            batch = inputs[0]
            workload["training_samples"] += len(batch.original_lengths)
            workload["training_tokens"] += sum(batch.original_lengths)
            workload["padded_tokens"] += batch.input_ids.numel()

    handles.append(native_model.register_forward_pre_hook(count_work))
    for key, (_, param, _) in selected.items():

        def capture(gradient, key=key):
            if torch.isfinite(gradient).all() and gradient.abs().sum().item() > 0:
                gradients[key] = True

        handles.append(param.register_hook(capture))
    activation_dtypes = set()
    loss_components = {}
    early_learning_checked = False
    for module in native_model.encoder.modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, value: activation_dtypes.add(str(value.dtype))
                )
            )
            break

    started = time.monotonic()

    def observe_loss(_module, _inputs, value):
        nonlocal early_learning_checked
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("Training wall-clock budget exhausted")
        if _module.training and trainer.global_step >= 50 and not early_learning_checked:
            evidence = {
                key: {
                    "changed": key in selected and selected[key][2] != parameter_hash(selected[key][1]),
                    "nonzero_finite_gradient": gradients.get(key, False),
                }
                for key in evidence_groups
            }
            Path(output, "early-learning-evidence.json").write_text(
                dumps({"step": trainer.global_step, "parameters": evidence}, indent=2)
            )
            if not all(v["changed"] and v["nonzero_finite_gradient"] for v in evidence.values()):
                raise RuntimeError("Early learning check failed; see early-learning-evidence.json")
            early_learning_checked = True
        if len(loss_components) < 32 and value.losses:
            for key, loss in value.losses.items():
                if torch.is_tensor(loss) and loss.numel() == 1:
                    loss_components[key] = float(loss.detach())

    handles.append(native_model.register_forward_hook(observe_loss))
    native_train = read_rows(Path(data) / "train.native.jsonl")
    native_dev = read_rows(Path(data) / "dev.native.jsonl")
    trainer = ExtractorTrainer(native_model, config)
    if extraction_dev:
        from .document_training import dev_callback as extraction_callback

        if config.metric_for_best != 'eval_pipeline_selection_score' or not config.greater_is_better or config.eval_strategy == 'no':
            raise ValueError('Extraction checkpoint selection requires the declared domain Dev selection score, maximized')
        trainer.compute_metrics = extraction_callback(extraction_dev, quality_policy, output, lambda: trainer.global_step,
                                                      **dev_thresholds)
    if generalist_dev:
        from .full_training import dev_callback

        if config.metric_for_best != 'eval_general_min_score' or not config.greater_is_better or config.eval_strategy == 'no':
            raise ValueError('Generalist checkpoint selection requires the minimum Dev score, maximized')
        trainer.compute_metrics = dev_callback(generalist_dev, quality_policy, output, lambda: trainer.global_step)
    if str(trainer.device) != device and not (
        device == "cuda" and str(trainer.device).startswith("cuda")
    ):
        raise ValueError("Native trainer selected an unexpected device")
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        result = trainer.train(train_data=native_train, eval_data=native_dev)
    finally:
        for handle in handles:
            handle.remove()
    # Upstream uses infinity as the sentinel when evaluation is disabled.
    if result.get("best_metric") is not None and not math.isfinite(result["best_metric"]):
        if config.eval_strategy != "no":
            raise RuntimeError("Nonfinite evaluation metric")
        result["best_metric"] = None
    changes = {
        key: {
            "parameter": name,
            "changed": before != parameter_hash(param),
            "initial_parameter_sha256": before,
            "verification_scope": "full representative parameter",
            "nonzero_finite_gradient": gradients[key],
        }
        for key, (name, param, before) in selected.items()
    }
    report = {
        "result": result,
        "elapsed_seconds": time.monotonic() - started,
        "workload": workload,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "required_learning_groups": evidence_groups,
        "wall_clock_limit_seconds": max_seconds,
        "parameters": changes,
        "device": str(trainer.device),
        "dtype": str(next(trainer.model.parameters()).dtype),
        "encoder_linear_activation_dtypes": sorted(activation_dtypes),
        "observed_loss_components": loss_components,
        "config": config_data,
        "generalist_dev": generalist_dev,
        "extraction_dev": extraction_dev,
        "quality_policy": quality_policy if generalist_dev or extraction_dev else None,
        "base_model": str(Path(model).resolve()),
        "initial_weights_sha256": initial_weights_sha256,
        "source_sha256": source_hash.hexdigest(),
        "dataset_sha256": hashlib.sha256(Path(data, "train.native.jsonl").read_bytes()).hexdigest(),
        "versions": {
            k: importlib.metadata.version(k) for k in ["torch", "transformers", "gliner2"]
        },
        "resume": "Weights-only restart; optimizer/scheduler/RNG resume unsupported by pinned trainer.",
    }
    Path(output, "training-evidence.json").write_text(dumps(report, indent=2))
    for key in evidence_groups:
        if key not in changes or not changes[key]["changed"] or not changes[key]["nonzero_finite_gradient"]:
            raise RuntimeError(f"{key} learning not demonstrated")
    return report
