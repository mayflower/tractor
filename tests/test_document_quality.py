import pytest

from tractor.document_quality import evaluate


def test_missing_real_evidence_cannot_pass_domain_gate():
    policy = {'domains': ['invoices'], 'shapes': ['single_object'],
              'required_origins': ['real/human', 'synthetic/teacher'], 'minimum_exclusive': .7,
              'qualification': 'no compensation'}
    report = evaluate([], {}, policy)
    assert not report['passed']
    assert len(report['checks']) == 4
    assert all(c['score'] is None for c in report['checks'].values())


def test_extra_predictions_are_rejected():
    with pytest.raises(ValueError):
        evaluate([], {'unknown': {}}, {})
