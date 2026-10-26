"""Local GPU training diagnostic; outputs under reports, never a release benchmark."""


def main():
    import gc
    import json
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
    report = {
        "scope": "training forward/backward only; no inference latency benchmark",
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
    }

    from tractor.biqwen_attention import BIDIRECTIONAL, install_attention

    select_gdn(b, "fla")
    install_attention(m.backbone, BIDIRECTIONAL)
    m.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    report = {"scope": "actual 4B hybrid FLA backward; no optimizer updates", "cases": []}
    out = Path("reports/biqwen-correction-20261026/fast-hybrid-preflight.json")
    m.eval()
    inputs = m.tokenizer("Anna Müller ist für Rechnungen zuständig.", return_tensors="pt").to(
        "cuda"
    )
    changed = inputs["input_ids"].clone()
    changed[0, -1] = (changed[0, -1] + 1) % m.backbone.config.vocab_size
    with torch.inference_mode():
        a = m.backbone(**inputs, use_cache=False).last_hidden_state[0, 0].float()
        c = (
            m.backbone(input_ids=changed, attention_mask=inputs["attention_mask"], use_cache=False)
            .last_hidden_state[0, 0]
            .float()
        )
        diff = float((a - c).abs().max())
        assert torch.isfinite(a).all() and torch.isfinite(c).all() and diff > 0
    report["future_context_max_difference"] = diff
    m.train()
    for label, raw in stress:
        example = encode_example(raw, b.encoding, m.tokenizer, limits)
        m.zero_grad(set_to_none=True)
        m.training_observations = []
        torch.cuda.reset_peak_memory_stats()
        result = m(b.encoding.collate([example], m.tokenizer.pad_token_id), decode=False)
        result["loss"].backward()
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
        assert any(
            p.grad is not None and bool(p.grad.count_nonzero())
            for n, p in m.named_parameters()
            if "lora_B" in n
        )
        if raw["schema"]["tables"]:
            assert any(
                p.grad is not None and bool(p.grad.count_nonzero())
                for n, p in m.named_parameters()
                if n.startswith("assignment.interaction.")
            )
        row = {
            "case": label,
            "tokens": len(example["input_ids"]),
            "loss": float(result["loss"].detach()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        report["cases"].append(row)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        del result
        m.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "passed"
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
