from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tractor.biqwen import LEGACY_SCHEMA, TYPED_SCHEMA, load_runtime, schema_for
from tractor.biqwen_architecture import JointFieldAssignment, assignment_logits
from tractor.biqwen_training import LEGACY_RECIPE, RECIPE, accumulate_observations, load_checkpoint
from tractor.biqwen_training_data import encode_example, native_example
from tractor.jsonio import dumps
from tractor.types import ExtractRequest


def projections(d=2):
    previous = nn.Module()
    previous.anchor_proj = nn.Linear(d, d)
    previous.field_proj = nn.Linear(d, d)
    with torch.no_grad():
        for layer in [previous.anchor_proj, previous.field_proj]:
            layer.weight.copy_(torch.eye(d))
            layer.bias.zero_()
    return previous


def test_joint_head_preserves_initial_scores_and_can_reverse_ownership():
    head = JointFieldAssignment(projections())
    anchor, field = torch.randn(2), torch.randn(2)
    assert torch.equal(head.condition(anchor, field), anchor + field)
    head.condition(anchor, field).sum().backward()
    assert head.interaction[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        head.interaction[0].weight.zero_()
        head.interaction[0].bias.zero_()
        head.interaction[0].weight[0, 4] = 1  # anchor * field interaction
        head.interaction[-1].weight[0, 0] = 4
    model = SimpleNamespace(assignment=head, config=SimpleNamespace(head_dim=2),
                            absent_embedding=torch.zeros(2))
    pool = {'states': torch.tensor([[1., 0.]]), 'source_logits': torch.zeros(1)}
    owners = []
    for query in [torch.tensor([1., 0.]), torch.tensor([-1., 0.])]:
        margins = []
        for a in [torch.tensor([1., 0.]), torch.tensor([-1., 0.])]:
            scores = assignment_logits(model, a, query, pool)
            margins.append(scores[1] - scores[0])
        owners.append(int(torch.stack(margins).argmax()))
    assert owners == [0, 1]
    empty = {'states': torch.zeros(0, 2), 'source_logits': torch.zeros(0)}
    assert assignment_logits(model, anchor, field, empty).shape == (1,)


def test_initial_joint_logits_match_original_record_equations():
    torch.manual_seed(17)
    previous = projections(32)
    head = JointFieldAssignment(previous)
    anchor, query, null = torch.randn(3, 32)
    states, source = torch.randn(48, 32), torch.randn(48)
    # Same computation as the pinned GLiNER primitive plus BiQwen ABSENT/source.
    condition = previous.anchor_proj(anchor) + previous.field_proj(query)
    expected = torch.cat(((condition * null).sum().view(1) / 32**.5,
                          torch.einsum('d,cd->c', condition, states) / 32**.5 + source))
    model = SimpleNamespace(assignment=head, config=SimpleNamespace(head_dim=32), absent_embedding=null)
    actual = assignment_logits(model, anchor, query, {'states': states, 'source_logits': source})
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('metadata,trainable', [
    ({'format': RECIPE, 'architecture': 'unknown', 'schema_version': TYPED_SCHEMA}, False),
    ({'format': RECIPE, 'architecture': 'biqwen-joint-record-v2', 'schema_version': LEGACY_SCHEMA}, False),
    ({'format': LEGACY_RECIPE}, True),
])
def test_incompatible_checkpoint_contracts_fail_before_model_loading(tmp_path, metadata, trainable):
    (tmp_path / 'checkpoint.json').write_text(dumps(metadata))
    with pytest.raises(ValueError, match='architecture/schema|cannot resume'):
        load_checkpoint(tmp_path, trainable=trainable)


@pytest.mark.parametrize('spec', [
    {'type': kind} for kind in ['text', 'person', 'place', 'organization', 'phone',
                               'email', 'address', 'date', 'time', 'datetime', 'interval',
                               'duration', 'percent', 'number', 'iban', 'url', 'identifier']
] + [{'type': 'quantity', 'unit': 'mm'}, {'type': 'money', 'currency': 'EUR'},
     {'type': 'postcode', 'country': 'DE'}, {'type': 'list', 'separator': 'semicolon'}])
def test_declared_types_reach_neural_schema_without_guessed_context(spec):
    req = ExtractRequest(text='Anna Müller: 12 cm.', fields={'value': {'description': 'Wert', **spec}})
    description = schema_for(req, TYPED_SCHEMA)['fields'][0]['description']
    assert '\nType: ' + spec['type'] in description
    assert 'country' not in description or 'country' in spec
    assert schema_for(req, LEGACY_SCHEMA)['fields'][0]['description'] == 'Wert'


def test_type_and_normalization_context_are_distinct():
    def req(kind):
        return ExtractRequest(text='Anna in Berlin', fields={'value': {'type': kind, 'description': 'Kontakt'}})
    assert schema_for(req('person'), TYPED_SCHEMA) != schema_for(req('place'), TYPED_SCHEMA)
    request = ExtractRequest(text='Breite 12 cm.', fields={
        'width': {'type': 'quantity', 'description': 'Breite', 'unit': 'mm'}})
    description = schema_for(request, TYPED_SCHEMA)['fields'][0]['description']
    assert 'processing (not source spelling)' in description and 'retain source units' in description
    isbn = ExtractRequest(text='ISBN 0-306-40615-2', fields={
        'isbn': {'type': 'identifier', 'format': 'isbn13', 'description': 'ISBN'}})
    description = schema_for(isbn, TYPED_SCHEMA)['fields'][0]['description']
    assert 'source:' not in description and 'processing (not source spelling)' in description
    assert 'isbn13' in description
    with pytest.raises(ValueError, match='schema version'):
        schema_for(request, 'unknown')


@pytest.mark.parametrize('text,start,end', [('Breite 12 cm.', 7, 12), ('Width 12 cm.', 6, 11)])
def test_real_tokenizer_training_api_parity(text, start, end):
    path = Path('models/biqwen-extract-decide-4b').resolve()
    if not path.exists():
        pytest.skip('Pinned local tokenizer is not available')
    from transformers import AutoTokenizer

    _, _, encoding = load_runtime(path)
    tokenizer = AutoTokenizer.from_pretrained(path / 'model/tokenizer', local_files_only=True)
    req = ExtractRequest(text=text, fields={
        'width': {'type': 'quantity', 'description': 'Width / Breite', 'unit': 'mm'}})
    raw = native_example(req, [{'width': [{'start': start, 'end': end, 'text': text[start:end]}]}])
    limits = dict(max_length=4096, schema_prefix=True, max_span_tokens=48,
                  candidate_top_k=48, max_gold_spans=64, max_records=16)
    training = encode_example(raw, encoding, tokenizer, limits)
    inference = encoding.encode({'text': req.text, 'schema': schema_for(req, TYPED_SCHEMA)},
                                tokenizer, max_length=4096, schema_prefix=True)
    for key in ['input_ids', 'doc_positions', 'query_positions', 'offsets', 'schema']:
        assert training[key] == inference[key]
    assert inference['gold'] is None
    assert encoding.token_span_to_chars(text, training['offsets'],
                                       training['gold']['links']['f0'][0]['source']) == (start, end)


def test_diagnostics_preserve_numerators_and_denominators_across_resume():
    observation = {'identity': {'language': 'de', 'domain': 'contacts', 'form': 'records'},
                   'counts': {'requests': 1, 'proposal_hits': 2, 'proposal_total': 3,
                              'assignment_terms': 2, 'assignment_sum': .4}}
    groups = {}
    accumulate_observations(groups, [observation, observation])
    # This structure is checkpointed atomically with coverage and optimizer state.
    restored = {k: dict(v) for k, v in groups.items()}
    accumulate_observations(restored, [observation])
    assert restored['all']['requests'] == 3
    assert restored['language:de']['proposal_hits'] == 6
    assert restored['domain:contacts']['proposal_total'] == 9
    assert restored['form:records']['assignment_terms'] == 6
    assert restored['all']['assignment_sum'] == pytest.approx(1.2)
