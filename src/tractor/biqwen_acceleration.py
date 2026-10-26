"""Explicit, verified Gated DeltaNet kernel selection on an existing model.

No packing, weight changes or alterations to the pinned upstream bundle. Models
with different kernel choices must run in separate processes (upstream dispatch
also has module-level state).
"""

from .biqwen import load_runtime

_BOOTSTRAPS = {}


def select_gdn(backend, mode):
    if mode not in {'torch', 'fla'}:
        raise ValueError('Expected torch or fla')
    if mode == 'fla' and backend.device != 'cuda':
        raise ValueError('FLA requires CUDA')
    path = backend.path.resolve()
    if path not in _BOOTSTRAPS:
        _BOOTSTRAPS[path] = load_runtime(path)[0]
    bootstrap = _BOOTSTRAPS[path]
    bootstrap.configure_gdn(mode)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    count = 0
    for module in backend.model.backbone.modules():
        if isinstance(module, qwen.Qwen3_5GatedDeltaNet):
            if callable(module.__dict__.get("chunk_gated_delta_rule")):
                module.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
            count += 1
    if not count:
        raise ValueError('No Qwen3.5 DeltaNet layers found')
    report = bootstrap.verify_gdn_binding(backend.model.backbone)
    if report.get('status') == 'module_lookup':
        if getattr(qwen.torch_chunk_gated_delta_rule, '__tractor_backend__', None) != mode:
            raise ValueError('Module-level DeltaNet dispatch differs from requested backend')
        report['module_lookup_layers'] = count
    elif report.get('layers') != {mode: count}:
        raise ValueError('Actual DeltaNet kernel bindings differ from requested backend')
    backend.gdn_backend = mode
    return report
