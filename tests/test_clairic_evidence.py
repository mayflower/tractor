"""Evidence is preserved from occurrence to final value without changing native output."""
import sys
from types import SimpleNamespace

import pytest
from clairic_backend import DescriptionBackend, spans_of

from tractor.service import extract, extract_traced
from tractor.types import ExtractRequest


def traced(text, fields, answers, signals=None, **request):
    backend = DescriptionBackend(answers, signals)
    req = ExtractRequest(text=text, fields=fields, **request)
    result, trace = extract_traced(backend, req)
    assert result == extract(DescriptionBackend(answers, signals), req)  # Observation only.
    return result, trace, backend


def test_equivalent_quantities_are_one_value_with_two_occurrences():
    text = "Breite 12 cm (120 mm)."
    result, trace, _ = traced(text, {"w": {"type": "quantity", "description": "Breite", "unit": "mm"}},
                              {"Breite": ["12 cm", "120 mm"]})
    assert result == {"data": {"w": 120}, "issues": []}
    [value] = trace.field("w").values
    assert value.value == 120
    assert [text[o.start:o.end] for o in value.occurrences] == ["12 cm", "120 mm"]


def test_unparsable_earlier_candidate_never_evidences_a_later_value():
    text = "ca. 5 cm, then 7 cm"
    result, trace, _ = traced(text, {"w": {"type": "quantity", "description": "W", "unit": "mm"}},
                              {"W": ["5 cm", "7 cm"]})
    evidence = trace.field("w")
    assert result["data"]["w"] == 70
    assert [o.text for o in evidence.occurrences_for(70)] == ["7 cm"]
    assert [(r.occurrence.text, r.code) for r in evidence.rejected] == [("5 cm", "non_scalar_quantity")]
    assert "non_scalar_quantity" in evidence.issues


def test_ambiguous_scalar_stays_a_list_with_each_value_evidenced():
    text = "12 cm or 13 cm"
    result, trace, _ = traced(text, {"w": {"type": "quantity", "description": "W", "unit": "mm"}},
                              {"W": ["12 cm", "13 cm"]})
    assert result == {"data": {"w": [120, 130]}, "issues": [{"field": "w", "code": "ambiguous_value"}]}
    assert [(v.value, v.occurrences[0].text) for v in trace.field("w").values] == [(120, "12 cm"), (130, "13 cm")]


def test_same_text_at_different_positions_keeps_every_occurrence():
    text = "Ref AB-1. Again: AB-1."
    _, trace, _ = traced(text, {"r": {"type": "identifier", "description": "Ref"}}, {"Ref": ["AB-1"]})
    [value] = trace.field("r").values
    assert [o.span for o in value.occurrences] == [[4, 8], [17, 21]]


@pytest.mark.parametrize("text,value", [
    ("Name: 𝔄lpha 𝔅eta\r\nNext: x", "𝔄lpha 𝔅eta"),
    ("Name: Café Noir\r\nNext", "Café Noir"),
    ("\r\n\r\nName:\tA B\r\n", "A B"),
])
def test_offsets_are_code_point_indices_into_unmodified_text(text, value):
    result, trace, _ = traced(text, {"n": {"type": "text", "description": "Name"}}, {"Name": [value]})
    [occurrence] = trace.field("n").values[0].occurrences
    assert result["data"]["n"] == value
    assert text[occurrence.start:occurrence.end] == value == occurrence.text


def test_signals_are_carried_and_missing_signals_stay_missing():
    text = "Alpha and Beta"
    fields = {"n": {"type": "text", "description": "Name", "cardinality": "unique_list"}}
    _, trace, _ = traced(text, fields, {"Name": ["Alpha", "Beta"]},
                         {"Alpha": {"raw_source_logit": 2.5, "source_score": 0.92}})
    alpha, beta = trace.field("n").values
    assert dict(alpha.occurrences[0].signals) == {"raw_source_logit": 2.5, "source_score": 0.92}
    assert dict(beta.occurrences[0].signals) == {}
    _, trace, _ = traced(text, fields, {"Name": ["Alpha"]})
    assert dict(trace.field("n").values[0].occurrences[0].signals) == {}


def test_multiple_fields_use_one_backbone_pass_and_report_tokens():
    text = "Invoice INV-7 dated 2026-10-01 for 12 cm."
    fields = {"i": {"type": "identifier", "description": "Invoice"},
              "d": {"type": "date", "description": "Date"},
              "w": {"type": "quantity", "description": "Width", "unit": "mm"}}
    _, trace, backend = traced(text, fields, {"Invoice": ["INV-7"], "Date": ["2026-10-01"], "Width": ["12 cm"]})
    assert backend.encoder_calls == 1 and trace.backbone_calls == 1
    assert trace.input_tokens == backend.last_tokens > 0


def test_records_keep_anchor_identity_after_sorting():
    text = "B: 2 cm. A: 1 cm."

    class Records(DescriptionBackend):
        def candidates_with_signals(self, request):
            if request.text != text:
                return [], {}
            rows = [{"name": spans_of(text, "B"), "w": spans_of(text, "2 cm")},
                    {"name": spans_of(text, "A"), "w": spans_of(text, "1 cm")}]
            return rows, {((0, 1), "w", 3, 7): {"assignment_margin": 1.5}}

    req = ExtractRequest(text=text, many=True, anchor="name", fields={
        "name": {"type": "text", "description": "Name"},
        "w": {"type": "quantity", "description": "W", "unit": "mm"}})
    result, trace = extract_traced(Records({}), req)
    assert result["data"] == [{"name": "B", "w": 20}, {"name": "A", "w": 10}]
    assert dict(trace.field("w", 0).values[0].occurrences[0].signals) == {"assignment_margin": 1.5}
    assert trace.field("w", 1).values[0].occurrences[0].signals == {}


def test_biqwen_decoder_returns_raw_signals_of_selected_occurrences(monkeypatch):
    torch = pytest.importorskip("torch")
    from tractor.biqwen import decode_multiple, schema_for

    monkeypatch.setitem(sys.modules, "tractor._biqwen_runtime.encoding",
                        SimpleNamespace(token_span_to_chars=lambda text, offsets, pair: pair))
    req = ExtractRequest(text="Alpha 12 cm", fields={"n": {"type": "text", "description": "Name"}})
    pools = {"f0": {"pairs": [(0, 5), (6, 11)], "source_logits": torch.tensor([3., -3.])}}
    model = SimpleNamespace(config=SimpleNamespace(source_threshold=.5, candidate_top_k=48),
                            _temperature=lambda _: 2.)
    prediction = decode_multiple(model, {"text": req.text, "offsets": [], "schema": schema_for(req)},
                                 pools, torch.zeros(1), None, None)
    assert prediction["fields"]["f0"] == [[0, 5]]
    [signal] = prediction["signals"]
    assert signal["raw_source_logit"] == 3.0 and signal["source_temperature"] == 2.0
    assert signal["source_score"] == pytest.approx(float(torch.sigmoid(torch.tensor(1.5))))
    assert "assignment_margin" not in signal


def test_biqwen_signals_map_to_public_fields_in_the_same_pass():
    pytest.importorskip("torch")
    from tractor.biqwen import BiQwenExtractor

    req = ExtractRequest(text="Alpha 12 cm", fields={"n": {"type": "text", "description": "Name"}})
    prediction = {"fields": {"f0": [[0, 5]]}, "tables": {},
                  "signals": [{"field": "f0", "span": [0, 5], "anchor": None, "raw_source_logit": 3.0,
                               "source_temperature": 1.0, "source_score": 0.95}]}

    class Model:
        backbone_forward_calls = 0
        tokenizer = SimpleNamespace(pad_token_id=0)
        config = SimpleNamespace(schema_prefix=True)

        def __call__(self, _):
            self.backbone_forward_calls += 1
            return {"predictions": [prediction]}

    backend = BiQwenExtractor.__new__(BiQwenExtractor)
    backend.input_limit, backend.model = 4096, Model()
    backend.encoding = SimpleNamespace(encode=lambda *a, **kw: {"input_ids": [1, 2, 3]}, collate=lambda *a: None)
    result, trace = extract_traced(backend, req)
    assert result["data"] == {"n": "Alpha"}
    assert trace.backbone_calls == 1 and trace.input_tokens == 3
    assert trace.field("n").values[0].occurrences[0].signals["raw_source_logit"] == 3.0
