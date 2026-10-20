"""Pinned native GLiNER integration. ML imports stay in this server-only module."""

import copy
import hashlib
import math
import os
import time
from functools import lru_cache
from pathlib import Path

from .jsonio import dumps, loads
from .types import ExtractRequest, Quantity, TractorError

BASE_MODEL = "fastino/gliner2.5-multi-v1"
REVISION = "cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3"
COMMIT = "b86d35c02f2c1a4d9840ea50197bd1b6d13b031e"


def load_native(path, device="cpu", precision="fp32"):
    """Shared offline loader for training and inference, independent of API policy."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from gliner2 import AutoExtractor

    path = Path(path).resolve()
    if not path.joinpath("model.safetensors").is_file():
        raise ValueError(f"Missing local model: {path}")
    model = AutoExtractor.from_pretrained(str(path))
    model.to(device)
    if precision == "bf16":
        if device != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 requires supported CUDA hardware")
        model.bfloat16()
    elif precision == "fp16":
        if device != "cuda":
            raise ValueError("FP16 requires CUDA")
        model.half()
    elif precision == "fp32":
        model.float()
    else:
        raise ValueError("Unsupported precision")
    return model


def prepare(output):
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL, revision=REVISION, local_dir=output)
    Path(output, "source-lock.json").write_text(
        dumps({"base_model": BASE_MODEL, "base_model_revision": REVISION, "gliner_commit": COMMIT})
    )


def field_ids(request):
    return {name: f"f{i}" for i, name in enumerate(request.fields)}


def description(field):
    return field.description + (
        "; vollständige Mengenangabe inklusive Einheit und Einschränkungen"
        if isinstance(field, Quantity)
        else ""
    )


@lru_cache(maxsize=256)
def _compile(fields_json, anchor, thresholds_json, _revision):
    from gliner2.inference.schema import Schema

    request = ExtractRequest(text="schema", fields=loads(fields_json), anchor=anchor)
    thresholds = loads(thresholds_json)
    ids = field_ids(request)
    schema = Schema()
    if anchor:
        builder = schema.structure("record", mode="natural", anchor=ids[anchor])
        for name, field in request.fields.items():
            # Cardinality concerns normalized VALUES, not evidence occurrences.
            # Retain all evidence so duplicate mentions can normalize together
            # and competing values remain detectable. Native exclusive ownership
            # assigns a scalar field's occurrence to its strongest record, rather
            # than returning that same occurrence for several different anchors.
            scalar = field.cardinality in {'required_one', 'optional_one'}
            builder.field(
                ids[name],
                dtype="list",
                description=description(field),
                cardinality="required_one" if name == anchor else "zero_or_more",
                exclusive=scalar,
                threshold=float(thresholds["anchor"] if name == anchor else thresholds.get(field.type, thresholds["text"])),
            )
        builder.build()
    else:
        schema.entities(
            {
                ids[name]: {
                    "description": description(field),
                    "dtype": "list",
                    "threshold": float(thresholds.get(field.type, thresholds["text"])),
                }
                for name, field in request.fields.items()
            }
        )
    return schema


def schema_for(request, thresholds=None, revision=REVISION):
    thresholds = thresholds or {"text": 0.5, "quantity": 0.5, "anchor": 0.5}
    # Only immutable, text-free serialization is cached; each caller owns its schema.
    return copy.deepcopy(
        _compile(
            dumps({k: v.model_dump() for k, v in request.fields.items()}),
            request.anchor,
            dumps(thresholds),
            revision,
        )
    )


def native_example(row):
    request = ExtractRequest.model_validate(row["request"])
    schema = schema_for(request).build()
    ids = field_ids(request)
    records = row["gold"]["records"]
    if request.anchor:
        # Empty list is an explicit negative record task with declared descriptions/metadata.
        schema["json_structures"] = [
            {"record": {ids[name]: spans for name, spans in record.items()}} for record in records
        ] or [{"record": {ids[name]: "" for name in request.fields}}]
    else:
        schema["entities"] = {ids[name]: records[0][name] for name in request.fields}
    return {"input": request.text, "output": schema}


class Extractor:
    def __init__(self, path, device="cpu", precision="fp32"):
        path = Path(path).resolve()
        if not path.joinpath("model.safetensors").exists():
            raise ValueError(f"Missing local model: {path}")
        self.path = path
        with path.joinpath("model.safetensors").open("rb") as weights:
            self.model_revision = hashlib.file_digest(weights, "sha256").hexdigest()
        self.model = load_native(path, device, precision)
        if self.model.architecture != "boundary" or not self.model.boundary_settings.enable_records:
            raise ValueError("Expected boundary architecture with trained record support")
        self.model.eval()
        self.device = device
        self.precision = precision
        # Conservative joint schema/text window from the actual checkpoint and encoder.
        self.input_limit = min(
            self.model.config.max_len, self.model.encoder.config.max_position_embeddings
        )
        self.thresholds = {"text": 0.5, "quantity": 0.5, "anchor": 0.5}
        if path.joinpath("thresholds.json").exists():
            self.thresholds = loads(path.joinpath("thresholds.json").read_text())
        self.last_timings = {}
        self.last_tokens = 0
        self.native_threshold = 0.5
        self.use_native_policy = path.joinpath("native-policy.json").exists()
        self.native_record_threshold = None
        if path.joinpath("native-policy.json").exists():
            policy = loads(path.joinpath("native-policy.json").read_text())
            if (policy.get("format") != "tractor-native-threshold-v1"
                    or policy.get("model_sha256") != self.model_revision
                    or policy.get("gliner_commit") != COMMIT
                    or isinstance(policy.get("threshold"), bool)
                    or not 0 < float(policy.get("threshold", 0)) <= 1):
                raise ValueError("Native policy does not match this model and implementation")
            self.native_threshold = float(policy["threshold"])
            if policy.get('record_threshold') is not None:
                value = policy['record_threshold']
                if isinstance(value, bool) or not 0 < float(value) <= 1:
                    raise ValueError('Invalid native record threshold')
                self.native_record_threshold = float(value)
        if set(self.thresholds) != {"text", "quantity", "anchor"} or any(
            isinstance(v, bool) or not 0 < float(v) <= 1 for v in self.thresholds.values()
        ):
            raise ValueError("Invalid bundle thresholds")
        if path.joinpath("normalization.json").exists():
            from .normalize import compatible_policy

            policy = loads(path.joinpath("normalization.json").read_text())
            if not compatible_policy(policy):
                raise ValueError("Bundle normalization contract differs from installed code")
        self.encoder_calls = 0
        self.model.encoder.register_forward_pre_hook(self._count)

    @classmethod
    def from_native(cls, model):
        """Borrow an in-training model using the production extraction path.

        Caller must remove borrowed_hook after evaluation; weights remain owned
        by the trainer. No second model load or alternate inference implementation.
        """
        instance = cls.__new__(cls)
        instance.model = model
        instance.input_limit = min(model.config.max_len, model.encoder.config.max_position_embeddings)
        instance.native_threshold = .5
        instance.use_native_policy = True
        instance.thresholds = {"text": .5, "quantity": .5, "anchor": .5}
        instance.model_revision = "training"
        instance.native_record_threshold = None
        instance.encoder_calls = 0
        instance.last_tokens = 0
        instance.last_timings = {}
        instance.borrowed_hook = model.encoder.register_forward_pre_hook(instance._count)
        return instance

    def _count(self, *_):
        self.encoder_calls += 1

    def preflight(self, request, schema):
        record = self.model.processor.transform_record(request.text, schema)
        if len(record.input_ids) > self.input_limit:
            raise TractorError(
                422, "token_limit", f"Combined schema/text exceeds {self.input_limit} tokens"
            )
        return len(record.input_ids)

    def candidates(self, request, hidden_states=None):
        thresholds = dict(self.thresholds)
        if getattr(self, "use_native_policy", False):
            threshold = self.native_record_threshold if request.anchor and getattr(self, "native_record_threshold", None) is not None else self.native_threshold
            thresholds = dict.fromkeys(("text", "quantity", "anchor"), threshold)
        schema = schema_for(request, thresholds, self.model_revision)
        started = time.perf_counter()
        self.last_tokens = self.preflight(request, schema)
        preprocessed = time.perf_counter()
        raw = self.model.batch_extract(
            [request.text],
            [schema],
            threshold=min(float(v) for v in thresholds.values()),
            include_spans=True,
            include_confidence=True,
            hidden_states=hidden_states,
        )[0]
        self.last_timings = {
            "preprocess": (preprocessed - started) * 1000,
            "inference": (time.perf_counter() - preprocessed) * 1000,
        }
        ids = field_ids(request)
        rows = raw.get("record", []) if request.anchor else [raw.get("entities", {})]
        if request.anchor and len(rows) >= self.model.boundary_settings.record_instance_queries:
            raise TractorError(422, "output_capacity", "Record capacity reached")
        records = []
        for row in rows:
            record = {}
            for name, internal in ids.items():
                values = row.get(internal, [])
                if values is None:
                    values = []
                values = [values] if isinstance(values, dict) else values
                if not isinstance(values, list):
                    raise RuntimeError("Unexpected native candidate representation")
                if len(values) >= self.model.boundary_settings.max_gold_per_query:
                    raise TractorError(422, "output_capacity", "Field candidate capacity reached")
                record[name] = values
            # Native formatting can filter an anchor after proposing its record.
            # A record without a surviving anchor is not a detected instance.
            if request.anchor and not record[request.anchor]:
                continue
            records.append(record)
        return records

    def extract_tasks(self, text, tasks, threshold=None):
        """Local native task interface. The HTTP Text/Quantity contract is unchanged."""
        from types import SimpleNamespace

        from .episodes import Task, compile_schema

        record_threshold = getattr(self, 'native_record_threshold', None) if threshold is None else None
        threshold = self.native_threshold if threshold is None else threshold
        if isinstance(threshold, bool) or not 0 < float(threshold) <= 1:
            raise ValueError("Invalid native threshold")
        tasks = [t if isinstance(t, Task) else Task.model_validate(t) for t in tasks]
        if record_threshold is not None and (isinstance(record_threshold, bool) or not 0 < float(record_threshold) <= 1):
            raise ValueError('Invalid native record threshold')
        schema = compile_schema(tasks, threshold, record_threshold)
        self.last_tokens = self.preflight(SimpleNamespace(text=text), schema)
        before = self.encoder_calls
        raw = self.model.batch_extract(
            [text], [schema], threshold=min(threshold, record_threshold) if record_threshold is not None else threshold,
            include_spans=True, include_confidence=True
        )[0]
        if self.encoder_calls - before != 1:
            raise RuntimeError("Native extraction must use one encoder invocation per window")
        def validate(value):
            if isinstance(value, dict):
                if "confidence" in value:
                    confidence = value["confidence"]
                    if (isinstance(confidence, bool) or not math.isfinite(float(confidence))
                            or not 0 <= float(confidence) <= 1):
                        raise RuntimeError("Invalid native confidence score")
                if {"start", "end", "text"} <= value.keys():
                    start, end = value["start"], value["end"]
                    if not (isinstance(start, int) and isinstance(end, int)
                            and 0 <= start < end <= len(text) and text[start:end] == value["text"]):
                        raise RuntimeError("Native output lost original evidence offsets")
                for child in value.values():
                    validate(child)
            elif isinstance(value, list):
                if (value and isinstance(value[0], dict) and "start" in value[0]
                        and len(value) >= self.model.boundary_settings.max_gold_per_query):
                    raise TractorError(422, "output_capacity", "Field candidate capacity reached")
                for child in value:
                    validate(child)
        validate(raw)
        for task in tasks:
            if task.kind == "records" and len(raw.get(task.id, [])) >= (
                self.model.boundary_settings.record_instance_queries
            ):
                raise TractorError(422, "output_capacity", "Record capacity reached")
        return raw

    def encode(self, request):
        import torch
        from gliner2.training.trainer import ExtractorCollator

        thresholds = dict(self.thresholds)
        if getattr(self, "use_native_policy", False):
            threshold = self.native_record_threshold if request.anchor and getattr(self, "native_record_threshold", None) is not None else self.native_threshold
            thresholds = dict.fromkeys(("text", "quantity", "anchor"), threshold)
        schema = schema_for(request, thresholds, self.model_revision)
        self.preflight(request, schema)
        batch = ExtractorCollator(self.model.processor, is_training=False, architecture="boundary")(
            [(request.text, schema.build())]
        )
        batch = batch.to(next(self.model.parameters()).device)
        with torch.inference_mode():
            return [self.model.encode_tokens(batch)[0]]
