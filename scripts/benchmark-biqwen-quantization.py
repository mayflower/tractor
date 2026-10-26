"""Local GPU training diagnostic; requires optional bitsandbytes==0.50.2.

Outputs under reports, never a release benchmark."""


def main():
    import gc
    import importlib.metadata
    import json
    import random
    import time
    from pathlib import Path

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModel, BitsAndBytesConfig

    from tractor.biqwen import BiQwenExtractor
    from tractor.biqwen_acceleration import select_gdn
    from tractor.biqwen_preflight import stress_examples
    from tractor.biqwen_training import TARGETS, configure_training_model
    from tractor.biqwen_training_data import encode_example

    out = Path("reports/biqwen-correction-20261026/quantization-speed.json")
    report = {
        "scope": "training forward/backward only; no optimizer steps or quality claims",
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "cases": [],
    }
    try:
        torch.manual_seed(42)
        b = BiQwenExtractor("models/biqwen-extract-decide-4b", "cuda", "bf16")
        del b.model.backbone
        gc.collect()
        torch.cuda.empty_cache()
        b.model.backbone = AutoModel.from_pretrained(
            b.path / "model/backbone",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        b.model.backbone = prepare_model_for_kbit_training(
            b.model.backbone, use_gradient_checkpointing=False
        )
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
        select_gdn(b, "fla")
        m.backbone.config.use_cache = False
        m.train()
        limits = json.loads((b.path / "model/extractor_config.json").read_text())["config"]
        from itertools import islice

        examples = [
            encode_example(r, b.encoding, m.tokenizer, limits)
            for _, r in islice(stress_examples(b), 3)
        ]
        for size in [2, 8, 16]:
            batch = b.encoding.collate(
                [examples[i % len(examples)] for i in range(size)], m.tokenizer.pad_token_id
            )
            durations = []
            losses = []
            for rep in range(4):
                m.zero_grad(set_to_none=True)
                m.training_observations = []
                torch.manual_seed(42)
                random.seed(42)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                output = m(batch, decode=False)
                output["loss"].backward()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if rep:
                    durations.append(elapsed)
                losses.append(float(output["loss"].detach()))
                assert all(
                    p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()
                )
                peak = torch.cuda.max_memory_allocated() / 2**30
                del output
            case = {
                "quantization": "NF4 double, BF16 compute",
                "kernel": "fla",
                "gradient_checkpointing": False,
                "batch_size": size,
                "seconds_per_microbatch": sum(durations) / len(durations),
                "requests_per_second": size / (sum(durations) / len(durations)),
                "peak_allocated_gib": peak,
                "losses": losses,
            }
            report["cases"].append(case)
            out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(case), flush=True)
            m.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=type(exc).__name__, message=str(exc))
        raise
    finally:
        out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
