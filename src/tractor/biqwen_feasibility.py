"""Explicitly bounded diagnostic, never a full-training or release recipe.

Matched causal/hybrid arms use one immutable selection and the same presentation
budget. The 16-document memorization experiment resets before the separate Dev
experiment; neither calibration nor test is opened.
"""

import argparse
import hashlib
import random
import time
from collections import Counter, defaultdict, deque
from itertools import batched
from pathlib import Path

from .biqwen import BiQwenExtractor
from .biqwen_acceleration import select_gdn
from .biqwen_attention import BIDIRECTIONAL, CAUSAL, attention_contract, install_attention
from .biqwen_training import (
    TARGETS,
    IndexedJSONL,
    configure_training_model,
    evaluate_dev,
    prepare_plan,
)
from .biqwen_training_data import encode_example
from .coverage_assembly import sha
from .episodes import Episode, text_key
from .generalist_data import read_episodes
from .jsonio import dumps, loads

SCOPE = 'bounded-feasibility-not-full-training-v1'


def balanced_indices(identities, count, seed):
    """Balance language/form, then domain/source, without inspecting outcomes."""
    buckets = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(identities):
        if row['language'] not in {'de', 'en'}:
            raise ValueError('Only DE/EN inputs are permitted')
        key = hashlib.sha256(f"{seed}:{row['id']}:{row.get('task', '')}".encode()).hexdigest()
        buckets[(row['language'], row['form'])][(row['domain'], row['source'])].append((key, i))
    queues = {}
    for group, domains in sorted(buckets.items()):
        queues[group] = deque(deque(i for _, i in sorted(rows)) for _, rows in sorted(domains.items()))
    result = []
    active = deque(queues.values())
    while active and len(result) < count:
        domains = active.popleft()
        rows = domains.popleft()
        result.append(rows.popleft())
        if rows:
            domains.append(rows)
        if domains:
            active.append(domains)
    if len(result) != count:
        raise ValueError('Insufficient distinct eligible examples for requested diagnostic')
    return result


def microbatches(indices, lengths, max_batch=8, token_budget=2048):
    """Keep every request, with independent padding; long requests run alone."""
    batch = []
    for i in indices:
        candidate = [*batch, i]
        if batch and (len(candidate) > max_batch or max(lengths[j] for j in candidate) * len(candidate) > token_budget):
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def write_json(path, value):
    pending = path.with_suffix(path.suffix + '.writing')
    pending.write_text(dumps(value, indent=2) + '\n')
    pending.replace(path)


def prepare(config_path, root):
    config = loads(Path(config_path).read_text())
    plan = prepare_plan(config)  # full source/model/compiler/split bindings remain enforced
    if root.exists():
        raise FileExistsError('Use a fresh diagnostic output directory')
    root.mkdir(parents=True)
    data = Path(config['dataset']['path'])
    identities = [loads(line) for line in (data / 'train.identities.jsonl').open()]
    selected = balanced_indices(identities, 512, config['seed'])
    dataset = IndexedJSONL(data / 'train.native.jsonl')
    native = [dataset[i] for i in selected]
    dataset.close()
    for row, index in zip(native, selected, strict=True):
        if row['identity'] != identities[index] or row['identity']['split'] != 'train':
            raise ValueError('Native request and source identity differ')
    manifest = loads((data / 'manifest.json').read_text())
    wanted = {row['identity']['id'] for row in native}
    train_episodes = []
    with (Path(manifest['source']) / 'train.episodes.jsonl').open() as stream:
        for line in stream:
            raw = loads(line)
            if raw['id'] in wanted:
                train_episodes.append(Episode.model_validate(raw))
    by_id = {row.id: row for row in train_episodes}
    if set(by_id) != wanted:
        raise ValueError('Missing selected training source document')
    eligible = [i for i, row in enumerate(native) if len(by_id[row['identity']['id']].tasks) == 1]
    learn_indices = [eligible[i] for i in balanced_indices([native[i]['identity'] for i in eligible], 16, config['seed'] + 1)]
    learn = [by_id[native[i]['identity']['id']] for i in learn_indices]
    dev_all = list(read_episodes(data / 'dev.episodes.jsonl'))
    dev_identities = [{'id': r.id, 'language': r.language, 'form': 'records' if any(t.kind == 'records' for t in r.tasks) else 'entities',
                       'domain': r.family, 'source': r.source} for r in dev_all]
    dev = [dev_all[i] for i in balanced_indices(dev_identities, 128, config['seed'])]
    train_groups = {v for r in train_episodes for v in (r.source_group, r.derivation_group) if v}
    dev_groups = {v for r in dev for v in (r.source_group, r.derivation_group) if v}
    if train_groups & dev_groups or {text_key(r.text) for r in train_episodes} & {text_key(r.text) for r in dev}:
        raise ValueError('Train/Dev leakage')
    for name, rows in [('train.native.jsonl', native), ('learn.episodes.jsonl', [r.model_dump() for r in learn]),
                       ('dev.episodes.jsonl', [r.model_dump() for r in dev])]:
        (root / name).write_text(''.join(dumps(row) + '\n' for row in rows))
    binding = {'scope': SCOPE, 'source_plan': plan, 'source_config': config,
               'train_source_indices': selected, 'learn_indices': learn_indices,
               'train_requests': len(native), 'dev_documents': len(dev), 'learn_documents': len(learn),
               'generalization_epochs': 2, 'learn_updates': 32, 'effective_batch_size': 16,
               'gdn_backend': 'fla', 'max_microbatch': 8, 'padded_token_budget': 2048,
               'checkpointing_above_tokens': 512,
               'selection': 'deterministic language/form then domain/source balance; diagnostic only',
               'train_distribution': {k: dict(Counter(row['identity'][k] for row in native))
                                      for k in ['language', 'form', 'domain', 'source']},
               'dev_distribution': {k: dict(Counter(row[k] for row in [dev_identities[i] for i in balanced_indices(dev_identities,128,config['seed'])]))
                                    for k in ['language', 'form', 'domain', 'source']},
               'files': {name: sha(root / name) for name in ['train.native.jsonl', 'learn.episodes.jsonl', 'dev.episodes.jsonl']},
               'code': {str(p): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    write_json(root / 'experiment.json', binding)
    return binding


def train_budget(backend, encoded, indices, epochs, steps, directory, config, binding):
    import torch

    model = backend.model
    parameters = [p for p in model.parameters() if p.requires_grad]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': head, 'lr': float(config['head_lr'])},
                                  {'params': backbone, 'lr': float(config['backbone_lr'])}],
                                 weight_decay=float(config['weight_decay']), fused=True)
    lengths = [len(row['input_ids']) for row in encoded]
    schedule = []
    for epoch in range(epochs):
        order = list(indices)
        random.Random(config['seed'] + epoch).shuffle(order)
        # Sorting within a fixed optimizer batch changes padding, not its members.
        schedule.extend(sorted(batch, key=lambda i: lengths[i]) for batch in batched(order, binding['effective_batch_size']))
    if steps is not None:
        schedule = [schedule[i % len(schedule)] for i in range(steps)]
    history = []
    directory.mkdir()
    model.train()
    started = time.perf_counter()
    for step, batch in enumerate(schedule):
        model.gold_injection_probability = 1 - .75 * step / max(1, len(schedule) - 1)
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for micro in microbatches(batch, lengths, binding['max_microbatch'], binding['padded_token_budget']):
            if max(lengths[i] for i in micro) > binding['checkpointing_above_tokens']:
                model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            else:
                model.backbone.gradient_checkpointing_disable()
            model.training_observations = []
            output = model(backend.encoding.collate([encoded[i] for i in micro], model.tokenizer.pad_token_id), decode=False)
            loss = output['loss'] * len(micro) / len(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; diagnostic aborted')
            loss.backward()
            total += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, float(config['max_grad_norm']), error_if_nonfinite=True)
        optimizer.step()
        history.append({'step': step + 1, 'loss': total, 'gradient_norm': float(norm), 'requests': len(batch)})
        if (step + 1) % 8 == 0 or step + 1 == len(schedule):
            progress = {'phase': directory.name, 'step': step + 1, 'total': len(schedule), 'loss': total,
                        'requests': sum(r['requests'] for r in history), 'elapsed_seconds': time.perf_counter() - started}
            write_json(directory / 'progress.json', progress)
            print(dumps(progress), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {'steps': len(schedule), 'request_presentations': sum(len(batch) for batch in schedule),
              'unique_requests': len(set(i for batch in schedule for i in batch)),
              'seconds': elapsed, 'requests_per_second': sum(len(batch) for batch in schedule) / elapsed,
              'history': history, 'schedule_sha256': hashlib.sha256(dumps(schedule).encode()).hexdigest()}
    write_json(directory / 'training.json', result)
    del optimizer
    model.zero_grad(set_to_none=True)
    return result


def run_arm(root, arm):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors.torch import save_file

    binding = loads((root / 'experiment.json').read_text())
    if binding['scope'] != SCOPE or arm not in {'causal', 'hybrid'}:
        raise ValueError('Wrong diagnostic scope or arm')
    for name, digest in {**binding['code'], **{str(root / n): d for n, d in binding['files'].items()}}.items():
        if sha(Path(name)) != digest:
            raise ValueError(f'Diagnostic input/code changed: {name}')
    directory = root / arm
    directory.mkdir()
    config = binding['source_config']
    status = {'scope': SCOPE, 'arm': arm, 'status': 'running', 'full_training': False}
    write_json(directory / 'status.json', status)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    backend = BiQwenExtractor(config['model'], 'cuda', 'bf16')
    backend.model.backbone = get_peft_model(backend.model.backbone, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=config['lora_rank'], lora_alpha=config['lora_alpha'],
        lora_dropout=float(config['lora_dropout']), target_modules=TARGETS, bias='none'))
    model = configure_training_model(backend, config['loss_weights'])
    install_attention(model.backbone, BIDIRECTIONAL if arm == 'hybrid' else CAUSAL)
    kernel = select_gdn(backend, binding['gdn_backend'])
    model.backbone.config.use_cache = False
    model.backbone.enable_input_require_grads()
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    initial_hash = hashlib.sha256(b''.join(p.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() for p in initial.values())).hexdigest()
    limits = loads(Path(config['model'], 'model/extractor_config.json').read_text())['config']
    encoded = [encode_example(loads(line), backend.encoding, model.tokenizer, limits)
               for line in (root / 'train.native.jsonl').open()]
    learn = list(read_episodes(root / 'learn.episodes.jsonl'))
    dev = list(read_episodes(root / 'dev.episodes.jsonl'))
    policy = loads(Path(config['policy']).read_text())
    try:
        for phase, rows in [('learn-before', learn), ('dev-before', dev)]:
            write_json(directory / f'{phase}.json', evaluate_dev(backend, rows, policy, directory / f'{phase}.predictions.jsonl'))
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])
        train_budget(backend, encoded, binding['learn_indices'], 1, binding['learn_updates'], directory / 'learn', config, binding)
        write_json(directory / 'learn-after.json', evaluate_dev(backend, learn, policy, directory / 'learn-after.predictions.jsonl'))
        # Separate generalization trial must not inherit memorization updates.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in initial:
                    p.copy_(initial[name])
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])
        training = train_budget(backend, encoded, list(range(len(encoded))), binding['generalization_epochs'], None,
                                directory / 'generalization', config, binding)
        write_json(directory / 'dev-after.json', evaluate_dev(backend, dev, policy, directory / 'dev-after.predictions.jsonl'))
        model.backbone.save_pretrained(directory / 'adapter', safe_serialization=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if not k.startswith('backbone.')},
                  str(directory / 'heads.safetensors'))
        status.update(status='completed', initial_trainable_sha256=initial_hash, attention=attention_contract(model.backbone),
                      kernel=kernel, training=training, experiment_sha256=sha(root / 'experiment.json'),
                      checkpoint_scope='diagnostic weights only; not a release or resumable full checkpoint')
    except BaseException as exc:
        status.update(status='failed', error=type(exc).__name__, message=str(exc))
        raise
    finally:
        write_json(directory / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'causal', 'hybrid'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', default='configs/biqwen-full.json')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.config, args.root)
    else:
        try:
            run_arm(args.root, args.action)
        except BaseException as exc:
            status = args.root / args.action / 'status.json'
            if status.exists():
                previous = loads(status.read_text())
                previous.update(status='failed', error=type(exc).__name__, message=str(exc))
                write_json(status, previous)
            raise


if __name__ == '__main__':
    main()
