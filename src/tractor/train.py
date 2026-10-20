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
from .training_adapter import (
    configure_complete_batches,
    configure_decoder_only,
    configure_frozen_encoder,
    configure_processor,
    configure_record_loss,
)


def parameter_hash(parameter):
    values = parameter.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


class LearningEvidence:
    """Check actual updates across each head, not one arbitrary first tensor."""

    def __init__(self, model):
        import torch

        self.selected = {}
        self.gradients = {}
        self.handles = []
        encoder_selected = False
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = name.split(".")[0]
            if group == "encoder":
                if encoder_selected or parameter.numel() <= 100:
                    continue
                encoder_selected = True
            self.selected[name] = (group, parameter, parameter_hash(parameter))
            self.gradients[name] = False

            def capture(gradient, name=name):
                if not self.gradients[name] and torch.isfinite(gradient).all() and gradient.abs().sum().item() > 0:
                    self.gradients[name] = True

            self.handles.append(parameter.register_hook(capture))

    def report(self):
        groups = {}
        for name, (group, parameter, before) in self.selected.items():
            item = {
                "changed": before != parameter_hash(parameter),
                "initial_parameter_sha256": before,
                "nonzero_finite_gradient": self.gradients[name],
            }
            groups.setdefault(group, {"parameters": {}})["parameters"][name] = item
        for group in groups.values():
            items = list(group["parameters"].values())
            group["changed"] = any(item["changed"] for item in items)
            group["nonzero_finite_gradient"] = any(item["nonzero_finite_gradient"] for item in items)
            group["learning_demonstrated"] = any(
                item["changed"] and item["nonzero_finite_gradient"] for item in items
            )
            group["verification_scope"] = "full tensors; all trainable head parameters, representative encoder parameter"
        return groups


def train(model, data, config_path, device="cuda"):
    manifest_path = Path(data, "manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("training_ready") is False:
            raise ValueError("Dataset is explicitly not training-ready; resolve its audit findings first")
    config = json.loads(Path(config_path).read_text())
    if binding := config.get('tractor_dataset'):
        from .training_corpus import verify_training_binding
        verify_training_binding(data, binding)
        if config.get('max_steps', -1) != -1 or config.get('early_stopping', False):
            raise ValueError('Full-dataset training requires complete epochs, not a step cap or early stopping')
        if config.get('tractor_max_seconds', 7200) is not None:
            raise ValueError('Full-dataset training cannot have a short wall-clock cutoff')
    lock_path = Path("runs/.training.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(model, data, config_path, device)


def _train(model, data, config_path, device):
    import torch
    from gliner2.training import ExtractorTrainer, TrainingConfig

    config_data = json.loads(Path(config_path).read_text())
    dataset_binding = config_data.pop("tractor_dataset", None)
    evidence_groups = config_data.pop("tractor_required_groups", ["encoder", "boundary_head", "record_decoder"])
    max_seconds = config_data.pop("tractor_max_seconds", 7200)
    generalist_dev = config_data.pop("tractor_generalist_dev", None)
    extraction_dev = config_data.pop("tractor_extraction_dev", None)
    dev_thresholds = config_data.pop("tractor_dev_thresholds", {})
    record_loss_weight = config_data.pop("tractor_record_loss_weight", None)
    decoder_only = config_data.pop("tractor_decoder_only", False)
    freeze_encoder = config_data.pop("tractor_freeze_encoder", False)
    regression_baseline = config_data.pop("tractor_regression_baseline", None)
    assignment_positive_weight = config_data.pop("tractor_assignment_positive_weight", None)
    if not isinstance(decoder_only, bool):
        raise ValueError('tractor_decoder_only must be boolean')
    if not isinstance(freeze_encoder, bool):
        raise ValueError('tractor_freeze_encoder must be boolean')
    if freeze_encoder and (decoder_only or config_data.get('use_lora')
                           or 'encoder' in evidence_groups
                           or not {'boundary_head', 'record_decoder'} <= set(evidence_groups)):
        raise ValueError('Frozen-encoder training requires both extraction heads, no encoder evidence, no decoder-only mode or LoRA')
    if decoder_only and (evidence_groups != ['record_decoder'] or config_data.get('use_lora')):
        raise ValueError('Decoder-only training requires record_decoder evidence and no LoRA')
    if assignment_positive_weight is not None and not decoder_only:
        raise ValueError('Assignment loss experiments require frozen encoder/span weights')
    if regression_baseline and not extraction_dev:
        raise ValueError('Regression guards require production pipeline Dev evaluation')
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
    if record_loss_weight is not None:
        configure_record_loss(native_model, record_loss_weight)
    decoder_handle = configure_decoder_only(native_model) if decoder_only else None
    if freeze_encoder:
        decoder_handle = configure_frozen_encoder(native_model)
    frozen_hashes = {name: parameter_hash(param) for name, param in native_model.named_parameters()
                     if (decoder_only or freeze_encoder) and not param.requires_grad}
    learning = LearningEvidence(native_model)
    handles = [decoder_handle] if decoder_handle else []
    handles.extend(learning.handles)
    workload = {"training_samples": 0, "training_tokens": 0, "padded_tokens": 0}

    def count_work(module, inputs):
        if module.training:
            batch = inputs[0]
            workload["training_samples"] += len(batch.original_lengths)
            workload["training_tokens"] += sum(batch.original_lengths)
            workload["padded_tokens"] += batch.input_ids.numel()

    handles.append(native_model.register_forward_pre_hook(count_work))
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
        if max_seconds is not None and time.monotonic() - started > max_seconds:
            raise TimeoutError("Training wall-clock budget exhausted")
        if _module.training and trainer.global_step >= 50 and not early_learning_checked:
            observed = learning.report()
            evidence = {key: observed.get(key, {"learning_demonstrated": False}) for key in evidence_groups}
            Path(output, "early-learning-evidence.json").write_text(
                dumps({"step": trainer.global_step, "parameters": evidence}, indent=2)
            )
            if not all(v["learning_demonstrated"] for v in evidence.values()):
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
    complete_batches = configure_complete_batches(trainer) if dataset_binding else None
    if extraction_dev:
        from .document_training import dev_callback as extraction_callback

        if config.metric_for_best != 'eval_pipeline_selection_score' or not config.greater_is_better or config.eval_strategy == 'no':
            raise ValueError('Extraction checkpoint selection requires the declared domain Dev selection score, maximized')
        trainer.compute_metrics = extraction_callback(extraction_dev, quality_policy, output, lambda: trainer.global_step,
                                                      regression_baseline=regression_baseline, **dev_thresholds)
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
    assignment_loss = None
    if assignment_positive_weight is not None:
        from .assignment_training import AssignmentLoss

        assignment_loss = AssignmentLoss(assignment_positive_weight)
        handles.append(assignment_loss)
    try:
        result = trainer.train(train_data=native_train, eval_data=native_dev)
    finally:
        for handle in handles:
            handle.remove()
    if extraction_dev and not result.get('eval_metrics_history'):
        raise RuntimeError('No production pipeline checkpoint evaluation ran; use an evaluation schedule that includes epoch boundaries')
    # Upstream uses infinity as the sentinel when evaluation is disabled.
    if result.get("best_metric") is not None and not math.isfinite(result["best_metric"]):
        if config.eval_strategy != "no":
            raise RuntimeError("Nonfinite evaluation metric")
        result["best_metric"] = None
    changes = learning.report()
    report = {
        "result": result,
        "elapsed_seconds": time.monotonic() - started,
        "workload": workload,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "required_learning_groups": evidence_groups,
        "wall_clock_limit_seconds": max_seconds,
        "dataset_binding": dataset_binding,
        "complete_batches": complete_batches,
        "parameters": changes,
        "device": str(trainer.device),
        "dtype": str(next(trainer.model.parameters()).dtype),
        "encoder_linear_activation_dtypes": sorted(activation_dtypes),
        "observed_loss_components": loss_components,
        "record_loss_weight": native_model.boundary_settings.record_loss_weight,
        "assignment_loss": assignment_loss.evidence() if assignment_loss else None,
        "decoder_only": decoder_only,
        "freeze_encoder": freeze_encoder,
        "regression_baseline": regression_baseline,
        "frozen_parameters_verified": len(frozen_hashes),
        "frozen_parameter_changes": [name for name, param in native_model.named_parameters()
                                     if name in frozen_hashes and frozen_hashes[name] != parameter_hash(param)],
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
    if report['frozen_parameter_changes']:
        raise RuntimeError('Frozen parameters changed during training')
    for key in evidence_groups:
        if key not in changes or not changes[key]["learning_demonstrated"]:
            raise RuntimeError(f"{key} learning not demonstrated")
    return report
