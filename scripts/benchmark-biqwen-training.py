"""Local GPU training diagnostic; outputs under reports, never a release benchmark."""


def main():
    import gc
    import json
    import random
    import time
    from pathlib import Path

    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    from tractor.biqwen import BiQwenExtractor
    from tractor.biqwen_acceleration import select_gdn
    from tractor.biqwen_preflight import stress_examples
    from tractor.biqwen_training import TARGETS, configure_training_model
    from tractor.biqwen_training_data import encode_example

    out = Path("reports/biqwen-correction-20261026/training-speed.json")
    torch.manual_seed(42)
    b = BiQwenExtractor("models/biqwen-extract-decide-4b", "cuda", "bf16")
    b.model.backbone = get_peft_model(
        b.model.backbone,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            target_modules=TARGETS,
            bias="none",
        ),
    )
    m = configure_training_model(b, {"boundary": 1, "source": 1, "assignment": 1})
    m.backbone.config.use_cache = False
    m.backbone.enable_input_require_grads()
    m.train()
    limits = json.loads(
        Path("models/biqwen-extract-decide-4b/model/extractor_config.json").read_text()
    )["config"]
    stress = list(stress_examples(b))
    examples = [encode_example(r, b.encoding, m.tokenizer, limits) for _, r in stress[:3]]
    report = {
        "scope": "training forward/backward only; no inference latency benchmark",
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
    }
    refs = {}
    for kernel, checkpoint, batch_size in [
        ("torch", True, 2),
        ("fla", True, 2),
        ("fla", False, 2),
        ("fla", False, 4),
        ("fla", False, 8),
    ]:
        select_gdn(b, kernel)
        if checkpoint:
            m.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            m.backbone.gradient_checkpointing_disable()
        durations = []
        losses = []
        checks = []
        for rep in range(4):
            m.zero_grad(set_to_none=True)
            m.training_observations = []
            torch.manual_seed(42)
            random.seed(42)
            batch = b.encoding.collate(
                [examples[i % len(examples)] for i in range(batch_size)], m.tokenizer.pad_token_id
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = m(batch, decode=False)
            result["loss"].backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            finite = all(
                p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()
            )
            if not finite:
                raise ValueError("nonfinite gradient")
            if rep:
                durations.append(elapsed)
            losses.append(float(result["loss"].detach()))
            if batch_size == 2 and rep == 1:
                gradients = {
                    n: p.grad.detach().float().cpu().clone()
                    for n, p in m.named_parameters()
                    if p.grad is not None
                    and (n.startswith("boundaries.") or "layers.0." in n and "lora_B" in n)
                }
                if kernel == "torch":
                    refs = gradients
                    ref_loss = losses[-1]
                else:
                    relative = (
                        (
                            sum((v - refs[n]).square().sum() for n, v in gradients.items())
                            / sum(v.square().sum() for v in refs.values()).clamp_min(1e-20)
                        )
                        .sqrt()
                        .item()
                    )
                    checks.append(
                        {
                            "gradient_relative_l2": relative,
                            "loss_absolute_difference": abs(losses[-1] - ref_loss),
                        }
                    )
                    if relative > 0.15 or abs(losses[-1] - ref_loss) > 0.03:
                        raise ValueError(checks)
            peak = torch.cuda.max_memory_allocated()
            del result
        case = {
            "kernel": kernel,
            "gradient_checkpointing": checkpoint,
            "batch_size": batch_size,
            "tokens": len(examples[0]["input_ids"]),
            "seconds_per_microbatch": sum(durations) / len(durations),
            "requests_per_second": batch_size / (sum(durations) / len(durations)),
            "peak_allocated_gib": peak / 2**30,
            "losses": losses,
            "gradient_checks": checks,
        }
        report["cases"].append(case)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    # Worst supported context must also fit. No model update is performed.
    select_gdn(b, "fla")
    m.backbone.gradient_checkpointing_disable()
    report["long_context"] = []
    for label, raw in stress[-2:]:
        example = encode_example(raw, b.encoding, m.tokenizer, limits)
        m.zero_grad(set_to_none=True)
        m.training_observations = []
        torch.cuda.reset_peak_memory_stats()
        try:
            result = m(b.encoding.collate([example] * 2, m.tokenizer.pad_token_id), decode=False)
            result["loss"].backward()
            case = {
                "case": label,
                "tokens": len(example["input_ids"]),
                "batch_size": 2,
                "passed": True,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            del result
        except torch.OutOfMemoryError:
            case = {"case": label, "batch_size": 2, "passed": False, "error": "out_of_memory"}
        report["long_context"].append(case)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)
        m.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "completed"
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
