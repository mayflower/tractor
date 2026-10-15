"""Pinned native GLiNER integration. ML imports stay in this server-only module."""

import copy
import hashlib
import os
import time
from functools import lru_cache
from pathlib import Path

from .jsonio import dumps, loads
from .types import ExtractRequest, Quantity, TractorError

BASE_MODEL = "fastino/gliner2.5-multi-v1"
REVISION = "cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3"
COMMIT = "b86d35c02f2c1a4d9840ea50197bd1b6d13b031e"


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
            builder.field(
                ids[name],
                dtype="list",
                description=description(field),
                cardinality="required_one" if name == anchor else "zero_or_more",
                threshold=float(thresholds["anchor" if name == anchor else field.type]),
            )
        builder.build()
    else:
        schema.entities(
            {
                ids[name]: {
                    "description": description(field),
                    "dtype": "list",
                    "threshold": float(thresholds[field.type]),
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
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from gliner2 import AutoExtractor

        path = Path(path).resolve()
        if not path.joinpath("model.safetensors").exists():
            raise ValueError(f"Missing local model: {path}")
        self.path = path
        with path.joinpath("model.safetensors").open("rb") as weights:
            self.model_revision = hashlib.file_digest(weights, "sha256").hexdigest()
        self.model = AutoExtractor.from_pretrained(str(path))
        if self.model.architecture != "boundary" or not self.model.boundary_settings.enable_records:
            raise ValueError("Expected boundary architecture with trained record support")
        self.model.to(device)
        if precision == "bf16":
            if device != "cuda" or not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 requires supported CUDA hardware")
            self.model.bfloat16()
        elif precision == "fp16":
            if device != "cuda":
                raise ValueError("FP16 requires CUDA")
            self.model.half()
        else:
            self.model.float()
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
        if set(self.thresholds) != {"text", "quantity", "anchor"} or any(
            isinstance(v, bool) or not 0 < float(v) <= 1 for v in self.thresholds.values()
        ):
            raise ValueError("Invalid bundle thresholds")
        if path.joinpath("normalization.json").exists():
            from .normalize import ALIASES, PINT_UNITS

            policy = loads(path.joinpath("normalization.json").read_text())
            if (
                policy.get("contract") != "tractor-v1"
                or policy.get("aliases") != ALIASES
                or policy.get("units") != PINT_UNITS
            ):
                raise ValueError("Bundle normalization contract differs from installed code")
        self.encoder_calls = 0
        self.model.encoder.register_forward_pre_hook(self._count)

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
        schema = schema_for(request, self.thresholds, self.model_revision)
        started = time.perf_counter()
        self.last_tokens = self.preflight(request, schema)
        preprocessed = time.perf_counter()
        raw = self.model.batch_extract(
            [request.text],
            [schema],
            threshold=float(self.thresholds["anchor"]) if request.anchor else 0.25,
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

    def encode(self, request):
        import torch
        from gliner2.training.trainer import ExtractorCollator

        schema = schema_for(request, self.thresholds, self.model_revision)
        self.preflight(request, schema)
        batch = ExtractorCollator(self.model.processor, is_training=False, architecture="boundary")(
            [(request.text, schema.build())]
        )
        batch = batch.to(next(self.model.parameters()).device)
        with torch.inference_mode():
            return [self.model.encode_tokens(batch)[0]]
