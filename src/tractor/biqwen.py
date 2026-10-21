"""Pinned BiQwen extraction backend; normalization remains in tractor.service.

The upstream record decoder chooses one candidate per cell. Our inference-only
decoder retains every accepted occurrence, assigning it to its strongest record.
These margins are provisional, not calibrated multi-label probabilities.
"""

import hashlib
import importlib
import importlib.util
import sys
import time
from pathlib import Path
from types import MethodType

from .jsonio import loads
from .model import description, field_ids
from .types import TractorError

LOCK = Path(__file__).with_name("biqwen-lock.json")
DECODER_VERSION = "biqwen-multiple-evidence-v1"


def verify_bundle(path):
    lock = loads(LOCK.read_text())
    for name, expected in lock["files"].items():
        with (path / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"BiQwen bundle differs from pinned revision: {name}")
    return lock


def load_runtime(path):
    """Import verified local code under a private namespace, without sys.path edits."""
    name = "tractor._biqwen_runtime"
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path / "tractor_qwen/__init__.py":
            raise ValueError("Another BiQwen runtime is already loaded")
    else:
        spec = importlib.util.spec_from_file_location(
            name, path / "tractor_qwen/__init__.py",
            submodule_search_locations=[str(path / "tractor_qwen")],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location(name + ".bootstrap", path / "bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    return (
        bootstrap,
        importlib.import_module(name + ".model"),
        importlib.import_module(name + ".encoding"),
    )


def schema_for(request):
    ids = field_ids(request)
    fields = []
    for name, field in request.fields.items():
        spec = {"id": ids[name], "kind": "span", "description": description(field),
                "multiple": not bool(request.anchor), "resolve_references": False}
        if request.anchor:
            spec["table_id"] = "record"
            if name == request.anchor:
                spec["kind"] = "anchor"
        fields.append(spec)
    tables = []
    if request.anchor:
        tables.append({"id": "record", "description": "Alle Einträge vom Typ record",
                       "anchor_field_id": ids[request.anchor],
                       "field_ids": [v for k, v in ids.items() if k != request.anchor]})
    return {"fields": fields, "tables": tables, "choices": []}


def assign_occurrences(margins, source_probabilities, threshold):
    """One owner per occurrence, any number of occurrences per owner.

    Margins compare each candidate against that record's learned ABSENT logit.
    Source acceptance is independent of record assignment. Never force a choice
    between competing values in a scalar field; normalization owns that decision.
    """
    rows = [[] for _ in margins]
    for column, probability in enumerate(source_probabilities):
        if probability < threshold or not margins:
            continue
        scores = [row[column] for row in margins]
        best = max(scores)
        if best <= 0:
            continue
        owners = [i for i, score in enumerate(scores) if score == best]
        if len(owners) != 1:
            raise TractorError(422, "ambiguous_record", "Candidate has equally scored record owners")
        rows[owners[0]].append(column)
    return rows


def decode_multiple(self, meta, pools, queries, global_anchor, options):
    """Decode extraction only, from model scores; never inspect annotations."""
    import torch

    from tractor._biqwen_runtime.encoding import token_span_to_chars

    if meta["schema"].get("choices"):
        raise ValueError("Tractor supports extraction only")
    fields = meta["schema"]["fields"]
    indexed = {f["id"]: i for i, f in enumerate(fields)}
    result = {"fields": {}, "tables": {}}

    def probabilities(pool, temperature="source"):
        return torch.sigmoid(pool["source_logits"].float() / self._temperature(temperature))

    def span(field_id, index):
        return list(token_span_to_chars(meta["text"], meta["offsets"], pools[field_id]["pairs"][index]))

    def selected(pool, threshold, temperature="source"):
        indices = (probabilities(pool, temperature) >= threshold).nonzero().flatten().tolist()
        if len(indices) >= self.config.candidate_top_k:
            raise TractorError(422, "output_capacity", "Field candidate capacity reached")
        return indices

    for field in fields:
        if "table_id" not in field:
            fid = field["id"]
            result["fields"][fid] = [span(fid, i) for i in selected(pools[fid], self.config.source_threshold)]
    for table in meta["schema"]["tables"]:
        aid = table["anchor_field_id"]
        anchor_pool = pools[aid]
        indices = selected(anchor_pool, self.config.anchor_threshold, "anchor")
        if len(indices) > self.config.max_records:
            raise TractorError(422, "output_capacity", "Record capacity exceeded")
        rows = [{"anchor": span(aid, i), "fields": {}} for i in indices]
        for fid in table["field_ids"]:
            pool = pools[fid]
            source = probabilities(pool).tolist()
            if sum(p >= self.config.source_threshold for p in source) >= self.config.candidate_top_k:
                raise TractorError(422, "output_capacity", "Field candidate capacity reached")
            margins = []
            for index in indices:
                logits = self._assignment_logits(anchor_pool["states"][index], queries[indexed[fid]], pool).float()
                margins.append((logits[1:] - logits[0]).tolist())
            for row, owned in zip(rows, assign_occurrences(margins, source, self.config.source_threshold), strict=True):
                row["fields"][fid] = [span(fid, i) for i in owned]
        result["tables"][table["id"]] = rows
    return result


class BiQwenExtractor:
    backend = DECODER_VERSION

    def __init__(self, path, device="cpu", precision="fp32"):
        import torch

        if precision not in {"fp32", "bf16"}:
            raise ValueError("BiQwen supports fp32 or bf16")
        if precision == "bf16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("BF16 requires supported CUDA hardware")
        self.path = Path(path).resolve()
        lock = verify_bundle(self.path)
        self.model_revision = lock["revision"]
        self.weights_hashes = {k: v for k, v in lock["files"].items() if k.endswith(".safetensors")}
        bootstrap, runtime, self.encoding = load_runtime(self.path)
        bootstrap.configure_gdn("torch")
        self.model = runtime.SinglePassExtractor.from_pretrained(
            self.path / "model", device=device,
            dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
        ).eval()
        bootstrap.verify_gdn_binding(self.model.backbone)
        self.model._decode_example = MethodType(decode_multiple, self.model)
        self.device, self.precision = device, precision
        self.input_limit = self.model.config.max_length
        self.last_tokens, self.last_timings = 0, {}

    @property
    def encoder_calls(self):
        return self.model.backbone_forward_calls

    def candidates(self, request):
        import torch

        started = time.perf_counter()
        try:
            encoded = self.encoding.encode(
                {"text": request.text, "schema": schema_for(request)}, self.model.tokenizer,
                self.input_limit, schema_prefix=self.model.config.schema_prefix,
            )
        except self.encoding.CapacityError as exc:
            raise TractorError(422, "token_limit", str(exc)) from exc
        self.last_tokens = len(encoded["input_ids"])
        preprocessed = time.perf_counter()
        before = self.encoder_calls
        with torch.inference_mode():
            raw = self.model(self.encoding.collate([encoded], self.model.tokenizer.pad_token_id))["predictions"][0]
        if self.encoder_calls != before + 1:
            raise RuntimeError("BiQwen must use one backbone pass per request")
        self.last_timings = {"preprocess": (preprocessed - started) * 1000,
                             "inference": (time.perf_counter() - preprocessed) * 1000}
        ids = field_ids(request)
        rows = raw["tables"]["record"] if request.anchor else [{"fields": raw["fields"]}]
        records = []
        for row in rows:
            record = {}
            for name, fid in ids.items():
                offsets = [row["anchor"]] if name == request.anchor else row["fields"].get(fid, [])
                record[name] = []
                for start, end in sorted(offsets):
                    if not (type(start) is int and type(end) is int and 0 <= start < end <= len(request.text)):
                        raise RuntimeError("Invalid BiQwen evidence offsets")
                    record[name].append({"text": request.text[start:end], "start": start, "end": end})
            records.append(record)
        return records
