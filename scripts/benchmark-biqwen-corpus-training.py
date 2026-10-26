"""Matched forward/backward throughput on 64 requests from the full epoch order."""


def main():
    import gc
    import json
    import random
    import time
    from itertools import batched
    from pathlib import Path

    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    from tractor.biqwen import BiQwenExtractor
    from tractor.biqwen_acceleration import select_gdn
    from tractor.biqwen_feasibility import microbatches
    from tractor.biqwen_training import (
        TARGETS,
        IndexedJSONL,
        configure_training_model,
        epoch_order,
    )
    from tractor.biqwen_training_data import encode_example

    torch.manual_seed(42)
    backend = BiQwenExtractor('models/biqwen-extract-decide-4b', 'cuda', 'bf16')
    backend.model.backbone = get_peft_model(backend.model.backbone, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=32, lora_alpha=64, lora_dropout=0., target_modules=TARGETS, bias='none'))
    model = configure_training_model(backend, {'boundary': 1., 'source': 1., 'assignment': 1.})
    model.backbone.enable_input_require_grads()
    model.backbone.config.use_cache = False
    model.train()
    limits = json.loads((backend.path / 'model/extractor_config.json').read_text())['config']
    dataset = IndexedJSONL('data/biqwen-full-de-en-v2/train.native.jsonl')
    indices = epoch_order(len(dataset), 42, 0)[:64]
    raw = [dataset[i] for i in indices]
    encoded = [encode_example(row, backend.encoding, model.tokenizer, limits) for row in raw]
    dataset.close()
    lengths = [len(row['input_ids']) for row in encoded]
    report = {'scope': '64 fixed requests, same BF16 weights/loss; forward/backward, no optimizer update or inference latency',
              'source_indices': indices, 'identities': [row['identity'] for row in raw], 'token_lengths': lengths, 'cases': []}
    out = Path('reports/biqwen-correction-20261026/corpus-training-speed.json')
    for mode in ['torch', 'fla']:
        select_gdn(backend, mode)
        durations = []
        for repeat in range(3):
            torch.manual_seed(42)
            random.seed(42)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for update in batched(range(64), 16):
                model.zero_grad(set_to_none=True)
                batches = (batched(update, 2) if mode == 'torch' else
                           microbatches(sorted(update, key=lambda i: lengths[i]), lengths))
                for batch in batches:
                    if mode == 'torch' or max(lengths[i] for i in batch) > 512:
                        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                    else:
                        model.backbone.gradient_checkpointing_disable()
                    model.training_observations = []
                    output = model(backend.encoding.collate([encoded[i] for i in batch], model.tokenizer.pad_token_id), decode=False)
                    (output['loss'] * len(batch) / len(update)).backward()
                    del output
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if repeat:
                durations.append(elapsed)
            print(json.dumps({'mode': mode, 'repeat': repeat, 'seconds': elapsed}), flush=True)
        report['cases'].append({'mode': mode, 'requests_per_second': 64 / (sum(durations) / len(durations)),
                                'durations_seconds': durations, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})
        out.write_text(json.dumps(report, indent=2) + '\n')
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    report['speedup'] = report['cases'][1]['requests_per_second'] / report['cases'][0]['requests_per_second']
    report['status'] = 'completed'
    out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
