"""Epoch-based BiQwen full training with durable, exact request coverage."""

import argparse
import hashlib
import math
import random
import shutil
from itertools import batched
from pathlib import Path
from types import MethodType

from .biqwen import LEGACY_SCHEMA, TYPED_SCHEMA, BiQwenExtractor
from .biqwen_architecture import ARCHITECTURE, LEGACY_ARCHITECTURE, install_architecture
from .biqwen_attention import BIDIRECTIONAL, CAUSAL, attention_contract, install_attention
from .biqwen_training_data import VERSION, encode_example
from .coverage_assembly import sha
from .document_training import regression_guard, selection_score
from .generalist_data import read_episodes
from .inference_context import EVALUATION_PRECISION, evaluation_precision
from .jsonio import dumps, loads
from .pipeline_evaluation import evaluate_pipeline
from .training_corpus import verify_training_binding

LEGACY_RECIPE = 'biqwen-full-epochs-multiple-assignment-v1'
RECIPE = 'biqwen-full-epochs-typed-joint-v2'
BIDIRECTIONAL_RECIPE = 'biqwen-full-epochs-hybrid-bidirectional-v3'
FAST_RECIPE = 'biqwen-full-epochs-fast-v4'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z',
           'in_proj_b', 'in_proj_a', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']


def multiple_losses(self, meta, pools, queries, global_anchor, marginal, sample_index,
                    options, totals, diagnostics):
    """Learn all positive owner/candidate margins, preserving proposal injection."""
    import torch
    from torch.nn import functional as F

    gold = meta['gold']
    if gold is None:
        return
    if meta['schema']['choices'] or any(f.get('resolve_references') for f in meta['schema']['fields']):
        raise ValueError('Extraction-only training cannot contain decisions or reference resolution')
    before = {name: len(terms) for name, terms in totals.items()}
    missing_before = {k: diagnostics[k] for k in ['record_anchor_gold_missing', 'assignment_gold_missing']}

    def bce(logits, target):
        weights = torch.where(target > 0, 1., .25)
        return (F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none') * weights).sum() / weights.sum().clamp_min(1)

    fields = {field['id']: (i, field) for i, field in enumerate(meta['schema']['fields'])}
    for fid, (index, _) in fields.items():
        pool = pools[fid]
        totals['boundary'].append(self._boundary_loss(meta, index, pool, marginal, sample_index))
        target = torch.tensor([pair in pool['gold_source'] for pair in pool['pairs']],
                              dtype=torch.float32, device=self.device)
        if target.numel():
            totals['source'].append(bce(pool['source_logits'], target))
    tables = {t['id']: t for t in meta['schema']['tables']}
    for record in gold['records']:
        table = tables[record['table_id']]
        anchor_pool = pools[table['anchor_field_id']]
        anchor_index = anchor_pool['index'].get(tuple(record['anchor']))
        if anchor_index is None:
            diagnostics['record_anchor_gold_missing'] += 1
            continue
        anchor = anchor_pool['states'][anchor_index]
        for fid in table['field_ids']:
            pool = pools[fid]
            positive = {tuple(pair) for pair in record['fields'][fid]}
            # Missing proposals must never turn present values into absent labels.
            if positive - set(pool['index']):
                diagnostics['assignment_gold_missing'] += 1
                continue
            if not pool['pairs']:
                continue
            logits = self._assignment_logits(anchor, queries[fields[fid][0]], pool).float()
            target = torch.tensor([pair in positive for pair in pool['pairs']],
                                  dtype=torch.float32, device=self.device)
            totals['assignment'].append(bce(logits[1:] - logits[0], target))
    if hasattr(self, 'training_observations'):
        counts = {'requests': 1, 'gold_records': len(gold['records']),
                  'gold_assignment_cells': sum(len(r['fields']) for r in gold['records'])}
        for key in ['proposal_hits', 'proposal_total', 'proposal_count', 'injected']:
            counts[key] = sum(pool.get(key, 0) for pool in pools.values())
        counts.update({k: diagnostics[k] - value for k, value in missing_before.items()})
        for name, terms in totals.items():
            current = terms[before[name]:]
            counts[name + '_terms'] = len(current)
            counts[name + '_sum'] = float(torch.stack(current).detach().sum()) if current else 0.
        self.training_observations.append({'identity': meta.get('identity', {}), 'counts': counts})


def accumulate_observations(groups, observations):
    """Resumable numerators/denominators; exact counts, never averages of averages."""
    for observation in observations:
        identity = observation['identity']
        keys = ['all'] + [f'{dimension}:{identity[dimension]}'
                          for dimension in ['language', 'domain', 'form', 'source'] if dimension in identity]
        for key in keys:
            target = groups.setdefault(key, {})
            for name, value in observation['counts'].items():
                target[name] = target.get(name, 0) + value


def configure_training_model(backend, loss_weights=None):
    model = install_architecture(backend)
    if loss_weights is not None:
        for name, value in loss_weights.items():
            setattr(model.config, name + '_loss_weight', float(value))
    model._losses_for_example = MethodType(multiple_losses, model)
    for name, parameter in model.named_parameters():
        if name.startswith(('ref_query.', 'ref_value.', 'ref_null.', 'choice_classifier.')):
            parameter.requires_grad_(False)
    model.temperatures = dict.fromkeys(model.temperatures, 1.)
    model.training_observations = []
    return model


class IndexedJSONL:
    """Random access without retaining the full corpus as Python objects."""
    def __init__(self, path):
        self.path, self.offsets = Path(path), []
        with self.path.open('rb') as stream:
            while True:
                offset = stream.tell()
                if not stream.readline():
                    break
                self.offsets.append(offset)
        self.stream = self.path.open('rb')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        self.stream.seek(self.offsets[index])
        return loads(self.stream.readline())

    def close(self):
        self.stream.close()


def epoch_order(count, seed, epoch):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def update_batches(order, batch_size, accumulation, cursor=0):
    """No dropped remainder and no crossing an epoch to fill a batch."""
    if cursor < 0 or cursor > len(order) or batch_size < 1 or accumulation < 1:
        raise ValueError('Invalid epoch cursor or batch configuration')
    yield from batched(batched(order[cursor:], batch_size), accumulation)


def record_coverage(coverage, indices):
    for index in indices:
        if coverage[index]:
            raise ValueError('Training request repeated within the epoch')
        coverage[index] = 1


def validate_config(config):
    required = {'recipe', 'backend', 'model', 'model_revision', 'dataset', 'output_dir', 'num_epochs',
                'batch_size', 'gradient_accumulation', 'seed', 'head_lr', 'backbone_lr', 'weight_decay',
                'max_grad_norm', 'lora_rank', 'lora_alpha', 'lora_dropout', 'gradient_checkpointing',
                'gold_injection_final', 'checkpoint_updates', 'policy', 'baseline', 'baseline_sha256',
                'sampling', 'quotas', 'max_steps', 'early_stopping', 'validation_limit', 'loss_weights'}
    if config.get('recipe') == FAST_RECIPE:
        required.update({'attention_mode', 'runtime'})
        if config.get('attention_mode') not in {CAUSAL, BIDIRECTIONAL}:
            raise ValueError('Fast recipe requires an explicit supported attention mode')
        runtime = config.get('runtime', {})
        if set(runtime) != {'gdn_backend', 'max_microbatch', 'padded_token_budget', 'checkpointing_above_tokens'}:
            raise ValueError('Fast recipe requires explicit runtime limits')
        if runtime['gdn_backend'] != 'fla' or any(type(runtime[k]) is not int or runtime[k] < 1 for k in runtime if k != 'gdn_backend'):
            raise ValueError('Invalid fast runtime')
        if not config.get('gradient_checkpointing'):
            raise ValueError('Fast recipe requires conditional gradient checkpointing for long inputs')
    if config.get('recipe') == BIDIRECTIONAL_RECIPE:
        required.add('attention_mode')
        if config.get('attention_mode') != BIDIRECTIONAL:
            raise ValueError('The bidirectional recipe requires its explicit attention contract')
    if (set(config) != required or config['recipe'] not in {RECIPE, BIDIRECTIONAL_RECIPE, FAST_RECIPE}
            or config['backend'] != 'biqwen'):
        raise ValueError('Expected the explicit BiQwen full-training recipe')
    if (config['sampling'] is not False or config['quotas'] is not False or config['max_steps'] is not None
            or config['early_stopping'] is not False or config['validation_limit'] is not None):
        raise ValueError('Full training forbids sampling, step limits, early stopping and Dev subsets')
    for key in ['num_epochs', 'batch_size', 'gradient_accumulation', 'checkpoint_updates', 'lora_rank', 'lora_alpha']:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ['head_lr', 'backbone_lr', 'weight_decay', 'max_grad_norm']:
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not 0 <= config['lora_dropout'] < 1 or not 0 <= config['gold_injection_final'] <= 1:
        raise ValueError('Invalid dropout or gold-injection probability')
    if set(config['loss_weights']) != {'boundary', 'source', 'assignment'} or any(
            not math.isfinite(v) or v <= 0 for v in config['loss_weights'].values()):
        raise ValueError('Explicit positive boundary/source/assignment loss weights required')


def prepare_plan(config):
    validate_config(config)
    manifest = verify_training_binding(config['dataset']['path'], config['dataset'])
    if manifest.get('format') != VERSION or manifest['model_revision'] != config['model_revision']:
        raise ValueError('Wrong backend format or tokenizer revision')
    for name, digest in manifest['compiler_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError(f'Compiler changed since full preflight: {name}')
    from .biqwen import verify_bundle
    lock = verify_bundle(Path(config['model']))
    if lock['revision'] != config['model_revision']:
        raise ValueError('Initial model differs from the pinned full-training plan')
    baseline = loads(Path(config['baseline']).read_text())
    if config['recipe'] == FAST_RECIPE and baseline.get('gdn_backend') != config['runtime']['gdn_backend']:
        raise ValueError('Re-score full baseline Dev with the same GDN backend before fast full training')
    if sha(Path(config['baseline'])) != config['baseline_sha256']:
        raise ValueError('Regression baseline changed')
    data = Path(config['dataset']['path'])
    if baseline['input_sha256'] != sha(data / 'dev.episodes.jsonl'):
        raise ValueError('Baseline and full Dev differ')
    if baseline.get('schema_version') != TYPED_SCHEMA or manifest.get('schema_version') != TYPED_SCHEMA:
        raise ValueError('Full training requires the same typed schema for data and baseline')
    model_config = loads(Path(config['model'], 'model/extractor_config.json').read_text())['config']
    if (baseline.get('policy_sha256') != sha(Path(config['policy']))
            or baseline.get('threshold') != model_config['source_threshold']
            or baseline.get('record_threshold') != model_config['anchor_threshold']
            or baseline.get('evaluation_precision') != {**EVALUATION_PRECISION,
                                                       'backbone_dtype': 'bf16', 'heads_dtype': 'fp32'}
            or baseline.get('weights_sha256') != {k: v for k, v in lock['files'].items()
                                                 if k.endswith('.safetensors')}):
        raise ValueError('Baseline policy, thresholds, precision or initial weights differ')
    count = manifest['native_requests']['train']
    updates = math.ceil(math.ceil(count / config['batch_size']) / config['gradient_accumulation'])
    fingerprint = hashlib.sha256(dumps(config, sort_keys=True).encode()).hexdigest()
    sources = {str(p): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    sources.update({p: sha(Path(p)) for p in ['uv.lock', 'configs/extraction-quality.json',
                                           'flake.lock', 'flake.nix', 'configs/libpostal-data.sha256']})
    return {'status': 'prepared', 'training_started': False, 'training_complete': False,
            'dataset_manifest_sha256': config['dataset']['manifest_sha256'], 'config_sha256': fingerprint,
            'dataset_files': manifest['files'],
            'source_sha256': sources, 'native_requests_per_epoch': count,
            'unique_train_episodes': manifest['unique_train_episodes'], 'languages': manifest['languages'],
            'num_epochs': config['num_epochs'], 'optimizer_updates_per_epoch': updates,
            'total_optimizer_updates': updates * config['num_epochs'],
            'total_request_presentations': count * config['num_epochs'],
            'full_dev_episodes': manifest['counts']['dev_input_episodes'],
            'sampling': False, 'step_limit': None, 'choice_training': False,
            'architecture': ARCHITECTURE, 'schema_version': TYPED_SCHEMA,
            'attention_mode': config.get('attention_mode', CAUSAL),
            'record_assignment': 'multi-positive candidate-versus-ABSENT BCE',
            'checkpoint_selection': 'complete epochs only; full normalized Dev pipeline and regression guard'}


def evaluate_dev(backend, rows, policy, output):
    from .biqwen_diagnostics import infer_with_diagnostics

    predictions = {}
    stages = {}
    backend.model.eval()
    with evaluation_precision(), Path(output).open('w') as stream, \
            Path(str(output) + '.stages.jsonl').open('w') as diagnostics:
        for index, row in enumerate(rows):
            result, trace = infer_with_diagnostics(backend, row)
            accumulate_observations(stages, [{'identity': {'language': row.language, 'domain': row.family,
                                                          'source': row.source}, 'counts': trace['counts']}])
            diagnostics.write(dumps({'id': row.id, **trace}) + '\n')
            predictions[row.id] = result
            stream.write(dumps({'id': row.id, **result}) + '\n')
            if (index + 1) % 100 == 0:
                stream.flush()
                print(dumps({'phase': 'full_dev', 'completed': index + 1, 'total': len(rows)}), flush=True)
    report = evaluate_pipeline(rows, predictions, policy)
    report['span_stage_diagnostics'] = stages
    report['span_stage_diagnostics_scope'] = 'Exact annotated spans, not final-value correctness or selection criteria'
    return report


def save_checkpoint(root, backend, optimizer, state, plan, config):
    import torch
    from safetensors.torch import save_file

    destination = root / 'checkpoints' / f"step-{state['step']:08d}"
    temporary = destination.with_name(destination.name + '.writing')
    if destination.exists() or temporary.exists():
        raise FileExistsError(f'Checkpoint already exists: {destination}')
    temporary.mkdir(parents=True)
    model = backend.model
    model.backbone.save_pretrained(temporary / 'adapter', safe_serialization=True)
    heads = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if not k.startswith('backbone.')}
    save_file(heads, str(temporary / 'heads.safetensors'))
    torch.save({**state, 'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
                'plan': plan}, temporary / 'training-state.pt')
    contract = attention_contract(model.backbone)
    if contract['mode'] != config.get('attention_mode', CAUSAL):
        raise ValueError('Checkpoint attention differs from the training recipe')
    metadata = {'format': config.get('recipe', RECIPE), 'base_model': str(Path(config['model']).resolve()),
                'attention': contract,
                'architecture': getattr(model, 'tractor_architecture', LEGACY_ARCHITECTURE),
                'schema_version': backend.schema_version,
                'base_revision': backend.model_revision, 'base_weights': backend.weights_hashes,
                'config': config, 'step': state['step'], 'epoch': state['epoch'], 'cursor': state['cursor'],
                'temperatures': model.temperatures,
                'files': {str(p.relative_to(temporary)): sha(p) for p in temporary.rglob('*') if p.is_file()}}
    (temporary / 'checkpoint.json').write_text(dumps(metadata, indent=2) + '\n')
    temporary.rename(destination)
    pointer = {'checkpoint': str(destination.relative_to(root)), 'step': state['step']}
    pending = root / 'latest.json.writing'
    pending.write_text(dumps(pointer, indent=2) + '\n')
    pending.replace(root / 'latest.json')
    return pointer


def load_checkpoint(path, device='cpu', precision='fp32', trainable=False):
    path = Path(path)
    metadata = loads((path / 'checkpoint.json').read_text())
    if metadata['format'] not in {LEGACY_RECIPE, RECIPE, BIDIRECTIONAL_RECIPE, FAST_RECIPE}:
        raise ValueError('Unsupported BiQwen training checkpoint')
    if metadata['format'] in {RECIPE, BIDIRECTIONAL_RECIPE, FAST_RECIPE} and (metadata.get('architecture') != ARCHITECTURE
                                       or metadata.get('schema_version') != TYPED_SCHEMA):
        raise ValueError('Checkpoint architecture/schema contract differs')
    if trainable and metadata['format'] == LEGACY_RECIPE:
        raise ValueError('Legacy checkpoints cannot resume the new full-training recipe')
    for name, digest in metadata['files'].items():
        if sha(path / name) != digest:
            raise ValueError(f'Checkpoint artifact changed: {name}')
    if metadata['format'] == FAST_RECIPE:
        validate_config(metadata['config'])
    backend = BiQwenExtractor(metadata['base_model'], device, precision)
    if backend.model_revision != metadata['base_revision'] or backend.weights_hashes != metadata['base_weights']:
        raise ValueError('Checkpoint requires its exact merged BiQwen base, not original Qwen weights')
    apply_checkpoint(backend, inference_contract(metadata), path / 'adapter', path / 'heads.safetensors',
                     device, trainable)
    if not trainable:
        backend.model_revision = 'local:' + sha(path / 'checkpoint.json')
        backend.weights_hashes = {**backend.weights_hashes,
                                  **{k: v for k, v in metadata['files'].items() if k.endswith('.safetensors')}}
    return backend


def inference_contract(metadata):
    """The checkpoint facts inference needs; no data paths or optimizer state."""
    config = metadata['config']
    return {'format': metadata['format'], 'architecture': metadata.get('architecture'),
            'schema_version': metadata.get('schema_version'),
            'attention_mode': config.get('attention_mode', CAUSAL),
            'gdn_backend': config.get('runtime', {}).get('gdn_backend'),
            'attention': metadata.get('attention', {'mode': CAUSAL}),
            'temperatures': metadata['temperatures']}


def apply_checkpoint(backend, contract, adapter, heads_file, device, trainable=False):
    """Install adapter, heads and attention of one verified checkpoint on its exact base."""
    from peft import PeftModel
    from safetensors.torch import load_file

    model = backend.model
    if contract['format'] in {RECIPE, BIDIRECTIONAL_RECIPE, FAST_RECIPE}:
        model = install_architecture(backend, contract['architecture'])
    else:
        backend.schema_version = LEGACY_SCHEMA
    model.backbone = PeftModel.from_pretrained(model.backbone, adapter, is_trainable=trainable)
    if contract['format'] == FAST_RECIPE:
        mode = contract['attention_mode']
        if device == 'cuda':
            from .biqwen_acceleration import select_gdn
            select_gdn(backend, contract['gdn_backend'])
    else:
        mode = BIDIRECTIONAL if contract['format'] == BIDIRECTIONAL_RECIPE else CAUSAL
    if contract['attention_mode'] != mode:
        raise ValueError('Checkpoint recipe and attention configuration disagree')
    install_attention(model.backbone, mode)
    if contract['attention'] != attention_contract(model.backbone):
        raise ValueError('Checkpoint attention contract differs from the runtime')
    heads = load_file(str(heads_file))
    if set(heads) != {k for k in model.state_dict() if not k.startswith('backbone.')}:
        raise ValueError('Extraction heads do not match the runtime')
    model.load_state_dict(heads, strict=False)
    model.temperatures = contract['temperatures']
    if not trainable:
        model.eval()
    return backend


def train(config, resume=False):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    plan = prepare_plan(config)
    # JSON numeric data deliberately uses Decimal in the extraction pipeline.
    # Optimizer/dropout hyperparameters must be Python floats for PyTorch/PEFT.
    numeric = {key: float(config[key]) for key in ['head_lr', 'backbone_lr', 'weight_decay',
               'max_grad_norm', 'lora_dropout', 'gold_injection_final']}
    active_path = Path('configs/active-training.json')
    active = loads(active_path.read_text())
    if active.get('status') == 'requires_revision':
        raise ValueError('Full-training preparation still requires revision')
    if (active.get('backend') != 'biqwen' or active['binding'] != config['dataset']
            or loads(Path(active['training_config']).read_text()) != config):
        raise ValueError('Activate this exact full-training recipe before starting')
    root = Path(config['output_dir'])
    if resume:
        pointer = loads((root / 'latest.json').read_text())
        checkpoint = root / pointer['checkpoint']
        backend = load_checkpoint(checkpoint, 'cuda', 'bf16', trainable=True)
        state = torch.load(checkpoint / 'training-state.pt', map_location='cpu', weights_only=True)
        if state.pop('plan') != plan:
            raise ValueError('Resume requires unchanged data, recipe and pipeline implementation')
    else:
        if root.exists():
            raise FileExistsError('A fresh full run requires a new output directory; resume explicitly')
        root.mkdir(parents=True)
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])
        backend = BiQwenExtractor(config['model'], 'cuda', 'bf16')
        backend.model.backbone = get_peft_model(backend.model.backbone, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=config['lora_rank'], lora_alpha=config['lora_alpha'],
            lora_dropout=numeric['lora_dropout'], target_modules=TARGETS, bias='none'))
        state = {'step': 0, 'epoch': 0, 'cursor': 0,
                 'coverage': bytearray(plan['native_requests_per_epoch']), 'best_metric': None,
                 'diagnostics': {}}
    model = configure_training_model(backend, config['loss_weights'])
    install_attention(model.backbone, config.get('attention_mode', CAUSAL))
    if config['recipe'] == FAST_RECIPE:
        from .biqwen_acceleration import select_gdn
        select_gdn(backend, config['runtime']['gdn_backend'])
    model.backbone.config.use_cache = False
    if config['gradient_checkpointing']:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.backbone.enable_input_require_grads()
    head = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if not head or not backbone:
        raise ValueError('Both extraction heads and backbone adapters must be trained')
    optimizer_options = {'fused': head[0].is_cuda} if config['recipe'] == FAST_RECIPE else {}
    optimizer = torch.optim.AdamW([{'params': head, 'lr': numeric['head_lr']},
                                  {'params': backbone, 'lr': numeric['backbone_lr']}],
                                 weight_decay=numeric['weight_decay'], **optimizer_options)
    if resume:
        optimizer.load_state_dict(state.pop('optimizer'))
        torch.set_rng_state(state.pop('rng'))
        torch.cuda.set_rng_state_all(state.pop('cuda_rng'))
        random.setstate(state.pop('python_rng'))
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    data = Path(config['dataset']['path'])
    dataset = IndexedJSONL(data / 'train.native.jsonl')
    if len(dataset) != plan['native_requests_per_epoch']:
        raise ValueError('Training dataset count changed')
    dev = list(read_episodes(data / 'dev.episodes.jsonl'))
    if len(dev) != plan['full_dev_episodes'] or any(e.split != 'dev' for e in dev):
        raise ValueError('Full Dev changed')
    policy, baseline = loads(Path(config['policy']).read_text()), loads(Path(config['baseline']).read_text())
    limits = loads(Path(config['model'], 'model/extractor_config.json').read_text())['config']
    (root / 'plan.json').write_text(dumps(plan, indent=2) + '\n')
    (root / 'config.json').write_text(dumps(config, indent=2) + '\n')
    status = {'status': 'running', 'training_started': True, 'training_complete': False}

    def persist_status():
        progress = {**status, 'step': state['step'], 'epoch': state['epoch'], 'cursor': state['cursor']}
        pending = root / 'status.json.writing'
        pending.write_text(dumps({**progress, 'plan': plan}, indent=2) + '\n')
        pending.replace(root / 'status.json')
        current = loads(active_path.read_text())
        if current.get('binding') != config['dataset'] or current.get('backend') != 'biqwen':
            raise ValueError('Active training selection changed during the run')
        current.update(progress)
        pending = active_path.with_suffix('.json.writing')
        pending.write_text(dumps(current, indent=2) + '\n')
        pending.replace(active_path)

    persist_status()
    try:
        if not resume:
            # Even an interruption before the first periodic checkpoint can resume.
            save_checkpoint(root, backend, optimizer, state, plan, config)
        while state['epoch'] < config['num_epochs']:
            epoch = state['epoch']
            order = epoch_order(len(dataset), config['seed'], epoch)
            coverage = state['coverage']
            if len(coverage) != len(dataset) or sum(coverage) != state['cursor']:
                raise ValueError('Resume coverage/cursor mismatch')
            model.train()
            for batches in update_batches(order, config['batch_size'], config['gradient_accumulation'], state['cursor']):
                progress = state['step'] / max(1, plan['total_optimizer_updates'] - 1)
                model.gold_injection_probability = 1 - progress * (1 - numeric['gold_injection_final'])
                factor = min(1., (state['step'] + 1) / max(1, int(plan['total_optimizer_updates'] * .03)))
                factor *= .1 + .9 * (1 + math.cos(math.pi * progress)) / 2
                for group, lr in zip(optimizer.param_groups, [numeric['head_lr'], numeric['backbone_lr']], strict=True):
                    group['lr'] = lr * factor
                optimizer.zero_grad(set_to_none=True)
                total_samples = sum(len(b) for b in batches)
                loss_total = 0.
                observations = []
                prepared = None
                if config['recipe'] == FAST_RECIPE:
                    from .biqwen_batching import batch_indices, checkpointing
                    # Repartition only within this optimizer update. Exact epoch
                    # membership, effective batch size and tail coverage persist.
                    prepared = {i: encode_example(dataset[i], backend.encoding, model.tokenizer, limits)
                                for batch in batches for i in batch}
                    lengths = {i: len(row['input_ids']) for i, row in prepared.items()}
                    batches = list(batch_indices(sorted(prepared, key=lengths.get), lengths, config['runtime']))
                for indices in batches:
                    examples = ([prepared[i] for i in indices] if prepared is not None else
                                [encode_example(dataset[i], backend.encoding, model.tokenizer, limits) for i in indices])
                    if prepared is not None:
                        checkpointing(model.backbone, max(len(row['input_ids']) for row in examples), config['runtime'])
                    model.training_observations = []
                    output = model(backend.encoding.collate(examples, model.tokenizer.pad_token_id), decode=False)
                    loss = output['loss'] * (len(indices) / total_samples)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss; no skipped batch')
                    loss.backward()
                    loss_total += float(loss.detach())
                    observations.extend(model.training_observations)
                torch.nn.utils.clip_grad_norm_([*head, *backbone], numeric['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                accumulate_observations(state['diagnostics'], observations)
                record_coverage(coverage, [i for batch in batches for i in batch])
                state['cursor'] += total_samples
                state['step'] += 1
                if state['step'] % 10 == 0:
                    print(dumps({'phase': 'train', 'epoch': epoch + 1, 'step': state['step'],
                                 'seen': state['cursor'], 'total': len(dataset), 'loss': loss_total,
                                 'gold_injection_probability': model.gold_injection_probability,
                                 'diagnostics': state['diagnostics'].get('all', {})}), flush=True)
                    pending = root / 'diagnostics.json.writing'
                    pending.write_text(dumps({'step': state['step'], 'epoch': epoch + 1,
                                              'groups': state['diagnostics']}, indent=2) + '\n')
                    pending.replace(root / 'diagnostics.json')
                    persist_status()
                if state['step'] % config['checkpoint_updates'] == 0 and state['cursor'] < len(dataset):
                    save_checkpoint(root, backend, optimizer, state, plan, config)
                    prune_checkpoints(root)
            if state['cursor'] != len(dataset) or not all(v == 1 for v in coverage):
                raise ValueError('Cannot complete an epoch without exact full request coverage')
            for name, digest in plan['dataset_files'].items():
                if sha(data / name) != digest:
                    raise ValueError(f'Dataset changed during training: {name}')
            for name, digest in plan['source_sha256'].items():
                if sha(Path(name)) != digest:
                    raise ValueError(f'Pipeline implementation changed during training: {name}')
            summary = {'epoch': epoch + 1, 'seen_requests': sum(coverage), 'expected_requests': len(dataset),
                       'all_requests_seen_exactly_once': True,
                       'identity_sha256': sha(data / 'train.identities.jsonl'),
                       'coverage_sha256': hashlib.sha256(coverage).hexdigest()}
            (root / f'epoch-{epoch + 1}-coverage.json').write_text(dumps(summary, indent=2) + '\n')
            (root / f'epoch-{epoch + 1}-diagnostics.json').write_text(dumps(state['diagnostics'], indent=2) + '\n')
            model.training_observations = []
            report = evaluate_dev(backend, dev, policy, root / f'epoch-{epoch + 1}-dev.predictions.jsonl')
            report['regression_guard'] = regression_guard(report, baseline)
            score = selection_score(report)
            improved = report['regression_guard']['passed'] and (state['best_metric'] is None or score > state['best_metric'])
            if improved:
                state['best_metric'] = score
            report.update(epoch=epoch + 1, selection_metric=score, selected=improved)
            (root / f'epoch-{epoch + 1}-dev.json').write_text(dumps(report, indent=2) + '\n')
            state.update(epoch=epoch + 1, cursor=0, coverage=bytearray(len(dataset)), diagnostics={})
            pointer = save_checkpoint(root, backend, optimizer, state, plan, config)
            if improved:
                pending = root / 'best.json.writing'
                pending.write_text(dumps({**pointer, 'selection_metric': score}, indent=2) + '\n')
                pending.replace(root / 'best.json')
            prune_checkpoints(root)
            persist_status()
        status.update(status='completed', training_complete=True)
    except BaseException:
        status['status'] = 'interrupted_or_failed'
        raise
    finally:
        persist_status()
        dataset.close()


def prune_checkpoints(root):
    protected = {loads((root / name).read_text())['checkpoint'] for name in ['latest.json', 'best.json']
                 if (root / name).exists()}
    paths = sorted((root / 'checkpoints').glob('step-[0-9]*'), reverse=True)
    for old in paths[2:]:
        if str(old.relative_to(root)) not in protected and (old / 'checkpoint.json').is_file():
            shutil.rmtree(old)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/biqwen-full.json')
    parser.add_argument('--train', action='store_true', help='Start the full GPU run; otherwise verify and print the plan')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = loads(Path(args.config).read_text())
    if args.train:
        train(config, resume=args.resume)
    else:
        if args.resume:
            parser.error('--resume requires --train')
        print(dumps(prepare_plan(config), indent=2))
