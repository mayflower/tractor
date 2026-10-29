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

from .jsonio import dumps, loads
from .model import field_ids
from .types import TractorError

LOCK = Path(__file__).with_name("biqwen-lock.json")
DECODER_VERSION = "biqwen-multiple-evidence-v1"
LEGACY_SCHEMA = "biqwen-descriptions-v1"
TYPED_SCHEMA = "biqwen-types-v2"


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


def field_description(field, request_locale=None):
    """Encode the declared contract, separating source selection from processing.

    No facts or context are inferred. Output transformations cannot require the
    source to already use the requested representation. Group is internal address
    assembly metadata, not a semantic type or a source-selection instruction.
    """
    options = field.model_dump(exclude={'description', 'type', 'group', 'cardinality'},
                               exclude_none=True, exclude_defaults=True)
    selection = {key: options.pop(key) for key in ['component'] if key in options}
    if request_locale is not None and 'locale' not in options:
        options['locale'] = request_locale
    result = field.description + '\nType: ' + field.type
    if selection:
        result += '; source: ' + dumps(selection, sort_keys=True)
    if options:
        result += '; processing (not source spelling): ' + dumps(options, sort_keys=True)
    if field.type == 'quantity':
        result += '; retain source units'
    return result


def schema_for(request, version=LEGACY_SCHEMA):
    """Preserve descriptions as in BiQwen training; types still select normalizers.

    GLiNER's quantity-description suffix is a model-specific training convention,
    not part of the public field contract or BiQwen's training input format.
    """
    if version not in {LEGACY_SCHEMA, TYPED_SCHEMA}:
        raise ValueError('Unsupported BiQwen schema version')
    ids = field_ids(request)
    fields = []
    for name, field in request.fields.items():
        description = field.description if version == LEGACY_SCHEMA else field_description(field, request.locale)
        spec = {"id": ids[name], "kind": "span", "description": description,
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
    # Raw signals of selected occurrences, from this same decode; diagnostics only.
    trace = getattr(self, "trace_signals", False)
    result = {"fields": {}, "tables": {}, "signals": []}
    cached = {}

    def probabilities(pool, temperature="source"):
        return torch.sigmoid(pool["source_logits"].float() / self._temperature(temperature))

    def host_scores(field_id, kind):
        # One transfer per pool and temperature, not one per selected occurrence.
        if (field_id, kind) not in cached:
            pool = pools[field_id]
            cached[field_id, kind] = (pool["source_logits"].float().tolist(),
                                      probabilities(pool, kind).tolist(), float(self._temperature(kind)))
        return cached[field_id, kind]

    def span(field_id, index):
        return list(token_span_to_chars(meta["text"], meta["offsets"], pools[field_id]["pairs"][index]))

    def selected(pool, threshold, temperature="source"):
        indices = (probabilities(pool, temperature) >= threshold).nonzero().flatten().tolist()
        if len(indices) >= self.config.candidate_top_k:
            raise TractorError(422, "output_capacity", "Field candidate capacity reached")
        return indices

    def signal(field_id, index, anchor=None, margin=None, kind="source"):
        if not trace:
            return
        logits, scores, temperature = host_scores(field_id, kind)
        entry = {"field": field_id, "span": span(field_id, index), "anchor": anchor,
                 "raw_source_logit": logits[index], "source_temperature": temperature,
                 "source_score": scores[index]}
        if margin is not None:
            entry["assignment_margin"] = margin
        result["signals"].append(entry)

    for field in fields:
        if "table_id" not in field:
            fid = field["id"]
            indices = selected(pools[fid], self.config.source_threshold)
            result["fields"][fid] = [span(fid, i) for i in indices]
            for i in indices:
                signal(fid, i)
    heads = getattr(self, "uncertainty_heads", None)
    if trace and heads is not None and heads.released:
        # Optional released heads read this pass's field queries; never a new backbone call.
        result["field_signals"] = []
        for field in fields:
            if "table_id" not in field:
                fid = field["id"]
                logits = heads(queries[indexed[fid]], pools[fid]["source_logits"])
                result["field_signals"].append({"field": fid, **{k: float(v.detach()) for k, v in logits.items()}})
    for table in meta["schema"]["tables"]:
        aid = table["anchor_field_id"]
        anchor_pool = pools[aid]
        indices = selected(anchor_pool, self.config.anchor_threshold, "anchor")
        if len(indices) > self.config.max_records:
            raise TractorError(422, "output_capacity", "Record capacity exceeded")
        rows = [{"anchor": span(aid, i), "fields": {}} for i in indices]
        for i in indices:
            signal(aid, i, span(aid, i), kind="anchor")
        for fid in table["field_ids"]:
            pool = pools[fid]
            source = probabilities(pool).tolist()
            if sum(p >= self.config.source_threshold for p in source) >= self.config.candidate_top_k:
                raise TractorError(422, "output_capacity", "Field candidate capacity reached")
            margins = []
            for index in indices:
                logits = self._assignment_logits(anchor_pool["states"][index], queries[indexed[fid]], pool).float()
                margins.append((logits[1:] - logits[0]).tolist())
            owners = assign_occurrences(margins, source, self.config.source_threshold)
            for position, (row, owned) in enumerate(zip(rows, owners, strict=True)):
                row["fields"][fid] = [span(fid, i) for i in owned]
                for i in owned:
                    signal(fid, i, row["anchor"], margins[position][i])
        result["tables"][table["id"]] = rows
    return result


class BiQwenExtractor:
    backend = DECODER_VERSION
    schema_version = LEGACY_SCHEMA

    def __init__(self, path, device="cpu", precision="fp32", verified_lock=None):
        import torch

        if precision not in {"fp32", "bf16"}:
            raise ValueError("BiQwen supports fp32 or bf16")
        if precision == "bf16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("BF16 requires supported CUDA hardware")
        self.path = Path(path).resolve()
        # verified_lock: the caller hashed exactly these base files against the lock.
        lock = verified_lock or verify_bundle(self.path)
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
        return self._decode(request, trace=False)[0]

    def candidates_with_signals(self, request):
        """Candidates and their raw scores from one backbone pass.

        Signals are keyed by (record anchor offsets or None, public field name,
        start, end). They are uncalibrated; absence of a signal stays absent.
        """
        return self._decode(request, trace=True)

    def _decode(self, request, trace):
        import torch

        started = time.perf_counter()
        try:
            encoded = self.encoding.encode(
                {"text": request.text, "schema": schema_for(request, self.schema_version)}, self.model.tokenizer,
                self.input_limit, schema_prefix=self.model.config.schema_prefix,
            )
        except self.encoding.CapacityError as exc:
            raise TractorError(422, "token_limit", str(exc)) from exc
        self.last_tokens = len(encoded["input_ids"])
        preprocessed = time.perf_counter()
        before = self.encoder_calls
        self.model.trace_signals = trace
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
        names = {fid: name for name, fid in ids.items()}
        signals = {}
        for entry in raw.get("signals", []):
            start, end = entry["span"]
            anchor = tuple(entry["anchor"]) if entry.get("anchor") is not None else None
            signals[(anchor, names[entry["field"]], start, end)] = {
                k: entry[k] for k in ("raw_source_logit", "source_score", "source_temperature",
                                      "assignment_margin") if k in entry}
        for entry in raw.get("field_signals", []):
            signals[(None, names[entry["field"]], None, None)] = {
                k: v for k, v in entry.items() if k != "field"}
        return records, signals
