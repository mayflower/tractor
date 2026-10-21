import sys
from types import SimpleNamespace

import pytest
import torch

from tractor.biqwen import (
    BiQwenExtractor,
    assign_occurrences,
    decode_multiple,
    schema_for,
    verify_bundle,
)
from tractor.jsonio import dumps
from tractor.service import extract
from tractor.types import ExtractRequest, TractorError


def test_multiple_values_and_exclusive_record_ownership():
    assert assign_occurrences([[4, 3, -2], [1, 2, 5]], [.9, .9, .9], .5) == [[0, 1], [2]]
    assert assign_occurrences([[4, -1, 0]], [.4, .9, .9], .5) == [[]]
    with pytest.raises(TractorError, match='equally scored'):
        assign_occurrences([[1], [1]], [.9], .5)


def test_altered_runtime_is_rejected_before_import(tmp_path, monkeypatch):
    import hashlib

    from tractor import biqwen

    code = tmp_path / 'bootstrap.py'
    code.write_text('raise AssertionError("must never execute")')
    lock = tmp_path / 'lock.json'
    lock.write_text(dumps({'files': {'bootstrap.py': hashlib.sha256(b'approved').hexdigest()}}))
    monkeypatch.setattr(biqwen, 'LOCK', lock)
    with pytest.raises(ValueError, match='pinned revision'):
        verify_bundle(tmp_path)


def test_local_bundle_selects_backend_without_changing_api(tmp_path, monkeypatch):
    from tractor import biqwen, model
    from tractor.backends import load_extractor

    monkeypatch.setattr(biqwen, 'BiQwenExtractor', lambda *args: ('biqwen', args))
    monkeypatch.setattr(model, 'Extractor', lambda *args: ('gliner', args))
    assert load_extractor(tmp_path)[0] == 'gliner'
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model/extractor_config.json').write_text('{}')
    assert load_extractor(tmp_path, 'cuda', 'bf16') == ('biqwen', (tmp_path, 'cuda', 'bf16'))


def request(cardinality='optional_one'):
    return ExtractRequest(text='Alpha: 10 cm, 12 cm. Beta: 20 cm.', many=True, anchor='name', fields={
        'name': {'type': 'text', 'description': 'Produkt'},
        'width': {'type': 'quantity', 'description': 'Breite', 'unit': 'mm', 'cardinality': cardinality},
    })


def test_schema_disables_indirection_and_preserves_all_entity_evidence():
    req = request()
    schema = schema_for(req)
    assert not schema['choices']
    assert not any(f['resolve_references'] for f in schema['fields'])
    assert schema['tables'][0]['anchor_field_id'] == 'f0'
    assert all(f['multiple'] for f in schema_for(req.model_copy(update={'anchor': None, 'many': False}))['fields'])


def test_record_decoder_keeps_competing_values_and_separate_records(monkeypatch):
    # Use fixed neural scores to test the changed decision rule, not model quality.
    monkeypatch.setitem(sys.modules, 'tractor._biqwen_runtime.encoding', SimpleNamespace(
        token_span_to_chars=lambda text, offsets, pair: pair,
    ))
    pools = {
        'f0': {'pairs': [(0, 5), (21, 25)], 'states': torch.tensor([0, 1]),
               'source_logits': torch.tensor([5., 5.])},
        'f1': {'pairs': [(7, 12), (14, 19), (27, 32)],
               'source_logits': torch.tensor([5., 5., 5.])},
    }
    model = SimpleNamespace(
        config=SimpleNamespace(source_threshold=.5, anchor_threshold=.5, candidate_top_k=48, max_records=16),
        _temperature=lambda _: 1.,
        _assignment_logits=lambda anchor, query, pool: torch.tensor(
            [0., 4., 3., -2.] if anchor == 0 else [0., 1., 2., 5.]),
    )
    req = request()
    prediction = decode_multiple(model, {'text': req.text, 'offsets': [], 'schema': schema_for(req)},
                                 pools, torch.zeros(2), None, None)
    assert prediction['tables']['record'][0]['fields']['f1'] == [[7, 12], [14, 19]]
    assert prediction['tables']['record'][1]['fields']['f1'] == [[27, 32]]
    model.config.max_records = 1
    with pytest.raises(TractorError, match='Record capacity'):
        decode_multiple(model, {'text': req.text, 'offsets': [], 'schema': schema_for(req)},
                        pools, torch.zeros(2), None, None)


@pytest.mark.parametrize('cardinality', ['optional_one', 'unique_list'])
def test_biqwen_candidates_reach_normalization_without_scalar_collapse(cardinality):
    req = request(cardinality)
    backend = BiQwenExtractor.__new__(BiQwenExtractor)
    backend.input_limit = 4096
    prediction = {'tables': {'record': [
        {'anchor': [0, 5], 'fields': {'f1': [[7, 12], [14, 19]]}},
        {'anchor': [21, 25], 'fields': {'f1': [[27, 32]]}},
    ]}}

    class Model:
        backbone_forward_calls = 0
        tokenizer = SimpleNamespace(pad_token_id=0)
        config = SimpleNamespace(schema_prefix=True)

        def __call__(self, _):
            self.backbone_forward_calls += 1
            return {'predictions': [prediction]}

    backend.model = Model()
    backend.encoding = SimpleNamespace(
        encode=lambda *a, **kw: {'input_ids': [1, 2]},
        collate=lambda *a: None,
    )
    output = extract(backend, req)
    assert output['data'][0]['width'] == [100, 120]
    assert output['data'][1]['width'] == ([200] if cardinality == 'unique_list' else 200)
    assert output['issues'] == ([] if cardinality == 'unique_list' else [
        {'field': 'width', 'index': 0, 'code': 'ambiguous_value'}])
    assert backend.encoder_calls == 1
