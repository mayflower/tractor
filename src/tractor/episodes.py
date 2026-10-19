"""Versioned, offset-preserving extraction episodes independent of the HTTP contract."""

import hashlib
import random
import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .field_contracts import Processing


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Span(Strict):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str


class Answer(Strict):
    status: Literal["present", "absent", "ambiguous", "unknown_annotation"]
    items: list[Span] = Field(default_factory=list)
    answer_alternatives: list[list[Span]] = Field(default_factory=list)
    label: str | None = None
    final_values: list[str] | None = None
    final_annotation_missing: bool = False
    final_alternatives: list[list[str]] = Field(default_factory=list)
    # Offline expected API behavior only; never sent to inference or native training.
    expected_issues: list[str] | None = Field(default=None, max_length=16)

    @model_validator(mode="after")
    def consistent(self):
        if self.expected_issues is not None:
            if (self.status not in {'present', 'absent'} or self.final_annotation_missing
                    or self.label is not None):
                raise ValueError('Expected issues require a complete extraction outcome annotation')
            if len(set(self.expected_issues)) != len(self.expected_issues) or any(
                    not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', code) for code in self.expected_issues):
                raise ValueError('Expected issue codes must be unique bounded identifiers')
            if self.status == 'present' and self.final_values is None and not self.final_alternatives:
                raise ValueError('Expected issues require independently annotated final values')
            if self.status == 'present' and 'not_found' in self.expected_issues:
                raise ValueError('Missing extraction is not correct rejection of present evidence')
        if self.status == 'present' and self.final_values == [] and not self.expected_issues:
            raise ValueError('Empty positive final target requires an expected rejection issue')
        if self.final_annotation_missing and (self.final_values is not None or self.final_alternatives):
            raise ValueError("Missing final annotation cannot contain final values")
        if self.final_values is not None and self.final_alternatives:
            raise ValueError("Use final_values OR final_alternatives")
        if self.status != "present" and (self.final_values or self.final_alternatives):
            raise ValueError("Only present answers may contain positive final targets")
        if self.status in {"absent", "unknown_annotation"} and (self.items or self.answer_alternatives or self.label):
            raise ValueError("Non-present answers cannot supply positive targets")
        if self.status == "ambiguous" and (self.answer_alternatives or self.label):
            raise ValueError("Ambiguity uses competing evidence items, not answer alternatives")
        if self.items and self.answer_alternatives:
            raise ValueError("Use items OR answer_alternatives")
        if self.status == "present" and not (self.items or self.answer_alternatives or self.label):
            raise ValueError("Present answer requires evidence or a class label")
        if any(not alternative for alternative in self.answer_alternatives):
            raise ValueError("Empty answer alternative")
        return self


class Task(Strict):
    id: str
    kind: Literal["entities", "records", "relations", "classification"]
    description: str = Field(min_length=1)
    fields: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    anchor: str | None = None
    cardinalities: dict[str, Literal["optional_one", "required_one", "zero_or_more", "one_or_more"]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid(self):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", self.id):
            raise ValueError("Invalid task id")
        if self.kind == "classification":
            if len(self.labels) < 2 or self.fields or self.anchor:
                raise ValueError("Classification requires at least two described labels")
        elif not self.fields or any(not value for value in self.fields.values()):
            raise ValueError("Extraction requires described fields")
        if self.cardinalities and (self.kind != "records" or not self.cardinalities.keys() <= self.fields.keys()):
            raise ValueError("Cardinalities require declared record fields")
        if self.kind == "records" and self.anchor not in self.fields:
            raise ValueError("Natural records require an explicit field anchor")
        if self.kind == "relations" and set(self.fields) != {"head", "tail"}:
            raise ValueError("Relations require head and tail")
        return self


class Gold(Strict):
    complete: bool
    records: list[dict[str, Answer]]


class Episode(Strict):
    version: Literal[1] = 1
    id: str
    source_group: str
    derivation_group: str
    source: str
    original_split: str
    split: Literal["train", "dev", "calibration", "test"]
    family: str
    language: str
    artifact: str
    annotation: Literal["synthetic_gold", "human", "silver"]
    text_origin: Literal["synthetic", "real", "unknown"] = "synthetic"
    label_origin: Literal["deterministic", "human", "teacher", "unknown"] = "deterministic"
    text: str = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1)
    gold: dict[str, Gold]
    processing: dict[str, dict[str, Processing]] = Field(default_factory=dict)
    locale: Literal['de-DE', 'en-US', 'en-GB'] | None = None

    @model_validator(mode="before")
    @classmethod
    def origin_defaults(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            annotation = value.get("annotation")
            value.setdefault("text_origin", "real" if annotation == "human" or (
                annotation == "silver" and value.get("source") != "gretel") else "synthetic")
            value.setdefault("label_origin", "human" if annotation == "human" else (
                "teacher" if annotation == "silver" else "deterministic"))
        return value

    @model_validator(mode="after")
    def valid(self):
        ids = [task.id for task in self.tasks]
        if len(set(ids)) != len(ids) or set(ids) != set(self.gold):
            raise ValueError("Task and gold IDs must match one-to-one")
        field_ids = [key for task in self.tasks if task.kind == "entities" for key in task.fields]
        if len(set(field_ids)) != len(field_ids):
            raise ValueError("Entity field keys must be unique across tasks")
        if not self.processing.keys() <= set(ids):
            raise ValueError("Processing references an undeclared task")
        for task in self.tasks:
            rules = self.processing.get(task.id, {})
            if not rules.keys() <= task.fields.keys():
                raise ValueError("Processing must reference declared fields")
            if task.anchor in rules and rules[task.anchor].kind not in {
                "literal", "person", "organization", "place", "identifier"
            }:
                raise ValueError("Record anchor must have a text, name, place or identifier type")
            gold = self.gold[task.id]
            keys = set(task.fields) if task.kind != "classification" else {"label"}
            if task.kind in {"entities", "classification"} and len(gold.records) != 1:
                raise ValueError("Document tasks require exactly one annotation record")
            for record in gold.records:
                if set(record) != keys:
                    raise ValueError("Gold keys must match task fields, including unknowns")
                for key, answer in record.items():
                    if key in rules and rules[key].kind != 'literal' and answer.status == 'present' and answer.final_values is None and not answer.final_alternatives and not answer.final_annotation_missing:
                        raise ValueError("Processed fields require independently annotated final values")
                    if answer.label is not None and answer.label not in task.labels:
                        raise ValueError("Gold label absent from schema")
                    for span in answer.items + [s for alt in answer.answer_alternatives for s in alt]:
                        if span.end <= span.start or self.text[span.start:span.end] != span.text:
                            raise ValueError("Gold span does not match original text")
        return self


def group_split(group, seed=42):
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 88 else "calibration" if bucket < 94 else "test"


def schema_variant(episode, seed):
    """Rename/permutate supported field IDs while preserving meaning, gold and grouping."""
    rng = random.Random(f"{seed}:{episode.id}")
    data = episode.model_dump()
    entity_counter = 0
    for task in data["tasks"]:
        if task["kind"] not in {"entities", "records"}:
            continue
        keys = list(task["fields"])
        rng.shuffle(keys)
        mapping = {key: f"f{entity_counter + i}" for i, key in enumerate(keys)}
        entity_counter += len(keys)
        rng.shuffle(keys)
        task["fields"] = {mapping[k]: task["fields"][k] for k in keys}
        if task['id'] in data['processing']:
            data['processing'][task['id']] = {mapping[k]: v for k, v in data['processing'][task['id']].items()}
        task["cardinalities"] = {mapping[k]: v for k, v in task["cardinalities"].items()}
        if task["anchor"]:
            task["anchor"] = mapping[task["anchor"]]
        data["gold"][task["id"]]["records"] = [
            {mapping[k]: record[k] for k in keys} for record in data["gold"][task["id"]]["records"]]
    data["id"] += f"-schema-{seed}"
    return Episode.model_validate(data)


def text_key(text):
    # Comparison only: never change the model input or its offsets.
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def audit_episodes(episodes):
    seen_ids, groups, texts, counts = set(), {}, {}, {}
    for episode in episodes:
        if episode.id in seen_ids:
            raise ValueError(f"Duplicate episode id: {episode.id}")
        seen_ids.add(episode.id)
        for group in (episode.source_group, episode.derivation_group):
            if group in groups and groups[group] != episode.split:
                raise ValueError("Source/derivation group crosses splits")
            groups[group] = episode.split
        key = text_key(episode.text)
        if key in texts and texts[key] != episode.split:
            raise ValueError("Equivalent source text crosses splits")
        texts[key] = episode.split
        counts[episode.split] = counts.get(episode.split, 0) + 1
    return {"episodes": counts, "unique_texts": len(texts), "group_keys": len(groups)}


def audit_similarity(episodes):
    """Approximate real-text overlap check; never alter inputs or use gold answers."""
    index, shingles_by_id, metadata, matches = defaultdict(set), {}, {}, []
    for episode in episodes:
        if episode.text_origin != "real":
            continue
        words = re.findall(r"\w+", episode.text.casefold())
        shingles = {" ".join(words[i:i + 5]) for i in range(len(words) - 4)}
        if len(shingles) < 10:
            continue
        signature = sorted(shingles, key=lambda s: hashlib.sha256(s.encode()).digest())[:8]
        candidates = set()
        for key in signature:
            if len(index[key]) < 100:
                candidates.update(index[key])
        for other in candidates:
            split, group = metadata[other]
            if split == episode.split or group == episode.source_group:
                continue
            overlap = len(shingles & shingles_by_id[other]) / min(len(shingles), len(shingles_by_id[other]))
            if overlap >= .9:
                matches.append({"a": other, "b": episode.id, "overlap": overlap})
        shingles_by_id[episode.id] = shingles
        metadata[episode.id] = (episode.split, episode.source_group)
        for key in signature:
            index[key].add(episode.id)
    return {"method": "8 minimum word-5-shingles; bucket cap 100; containment >= 0.9",
            "scope": "real texts; approximate detection is not proof of absence", "matches": matches}


def compile_schema(tasks, threshold=0.5, record_threshold=None):
    """Only task specifications enter inference; no gold argument is accepted."""
    from gliner2.inference.schema import Schema

    schema = Schema()
    for task in tasks:
        if task.kind == "entities":
            schema.entities({k: {"description": v, "dtype": "list", "threshold": threshold}
                             for k, v in task.fields.items()})
        elif task.kind == "records":
            builder = schema.structure(task.id, mode="natural", anchor=task.anchor)
            for key, description in task.fields.items():
                cardinality = task.cardinalities.get(key, "required_one" if key == task.anchor else "zero_or_more")
                builder.field(key, description=description, dtype="str" if cardinality in {"optional_one", "required_one"} else "list",
                              threshold=threshold if record_threshold is None else record_threshold, cardinality=cardinality,
                              exclusive=bool(task.cardinalities))
            builder.build()
        elif task.kind == "relations":
            schema.relations({task.id: task.description}, threshold=threshold)
        else:
            schema.classification(task.id, task.labels, prompt=task.description)
    return schema


def native_episode(episode):
    """Compile complete representable tasks; never turn missing annotation into negatives."""
    tasks, skipped = [], {}
    for task in episode.tasks:
        gold = episode.gold[task.id]
        states = {a.status for r in gold.records for a in r.values()}
        if not gold.complete or states & {"ambiguous", "unknown_annotation"}:
            skipped[task.id] = "incomplete_or_ambiguous"
        elif task.kind == "relations" and not gold.records:
            skipped[task.id] = "empty_structured_task_requires_collator_proof"
        else:
            tasks.append(task)
    if not tasks:
        return None, skipped
    output = compile_schema(tasks).build()
    output["json_structures"] = []
    output["relations"] = []
    for task in tasks:
        rows = episode.gold[task.id].records
        if task.kind == "classification":
            label = rows[0]["label"].label
            if label is None:
                raise ValueError("Classification supervision needs a declared label")
            next(c for c in output["classifications"] if c["task"] == task.id)["true_label"] = [label]
            continue
        targets = [{key: [s.model_dump() for s in (a.items or (a.answer_alternatives[0]
                    if a.answer_alternatives else []))] for key, a in row.items()} for row in rows]
        if task.kind == "entities":
            output["entities"].update(targets[0])
        else:
            field = "json_structures" if task.kind == "records" else "relations"
            output[field].extend({task.id: row} for row in targets)
            if task.kind == "records" and not targets:
                # Native schema declaration with no values; target reconstruction must
                # prove zero record instances, as for the existing product adapter.
                output[field].append({task.id: dict.fromkeys(task.fields, "")})
    return {"input": episode.text, "output": output}, skipped
