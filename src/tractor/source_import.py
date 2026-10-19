"""Explicit data-file adapters. No dataset loader code is executed."""

import ast
import hashlib
import re
from collections import Counter
from pathlib import Path

from .episodes import Episode, group_split
from .generalist_data import write_corpus
from .jsonio import dumps, loads


def unique_span(text, value):
    positions = [m.start() for m in re.finditer(re.escape(value), text)] if value else []
    if len(positions) != 1:
        raise ValueError("missing_or_ambiguous_occurrence")
    return {"start": positions[0], "end": positions[0] + len(value), "text": value}


def fetch_source(name, output, config="configs/generalist-sources.json"):
    """Explicit download of one pinned, non-executable training artifact."""
    import httpx
    from huggingface_hub import hf_hub_download

    spec = next(s for s in loads(Path(config).read_text())["sources"] if s["name"] == name)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    if "repo" in spec:
        path = Path(hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"],
                                   repo_type="dataset", local_dir=root))
    else:
        path = root / "train.json"
        with httpx.stream("GET", spec["url"], follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with path.open("xb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {**spec, "sha256": digest, "local_file": str(path)}
    root.joinpath("source.json").write_text(dumps(report, indent=2))
    return report


def envelope(source, ident, group, text, language, family, artifact, annotation, tasks, gold):
    split = group_split(group)
    return Episode.model_validate({
        "id": f"{source}:{ident}", "source": source, "source_group": group,
        "derivation_group": group, "original_split": "train", "split": split,
        "text": text, "language": language, "family": family, "artifact": artifact,
        "annotation": annotation, "tasks": tasks, "gold": gold,
        "text_origin": "synthetic" if source == "gretel" else "real",
        "label_origin": "human" if annotation == "human" else "teacher",
    })


def qa(row, source):
    text = row["context"]
    language = "de" if source == "germanquad" else "en"
    title = row.get("title") or text.splitlines()[0]
    group = "wikipedia:" + language + ":" + title.strip().replace("_", " ").casefold()
    alternatives = [[{"start": start, "end": start + len(value), "text": value}]
                    for start, value in zip(row["answers"]["answer_start"], row["answers"]["text"], strict=True)]
    answer = {"status": "present", "answer_alternatives": alternatives} if alternatives else {"status": "absent"}
    return envelope(source, row["id"], group, text, language, "question_answering", "encyclopedia", "human",
                    [{"id": "qa", "kind": "entities", "description": "Answer the question from the text",
                      "fields": {"answer": row["question"]}}],
                    {"qa": {"complete": True, "records": [{"answer": answer}]}})


def gretel(row):
    text = row["text"]
    fields, values = {}, {}
    # Published column contains Python literal lists, not JSON; literal_eval executes no calls.
    entities = ast.literal_eval(row["entities"]) if isinstance(row["entities"], str) else row["entities"]
    for entity in entities:
        span = unique_span(text, entity["entity"])
        for name in entity["types"]:
            fields[name] = name.replace("_", " ")
            values.setdefault(name, {"status": "present", "items": []})["items"].append(span)
    if not fields:
        raise ValueError("no_annotated_entities")
    # Dataset labels are synthetic annotations, not independently verified gold.
    return envelope("gretel", row["uid"], "gretel:" + row["uid"], text, "en", "pii", row["document_type"],
                    "silver", [{"id": "pii", "kind": "entities", "description": "Declared personal data",
                                "fields": fields}], {"pii": {"complete": True, "records": [values]}})


def parse_guidex_labels(value):
    """Parse literal constructor records without evaluating any expression."""
    tree = ast.parse(value, mode="eval").body
    if not isinstance(tree, ast.List):
        raise ValueError("labels_must_be_list")
    records = []
    for call in tree.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
            raise ValueError("unsupported_constructor")
        fields = {}
        for kw in call.keywords:
            if kw.arg is None or not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
                raise ValueError("non_string_or_nested_target")
            if kw.arg in fields:
                raise ValueError("duplicate_field")
            fields[kw.arg] = kw.value.value
        records.append((call.func.id, fields))
    return records


def guidex(row):
    text = row["unlabelled_sentence"]
    parsed = ast.parse(row["labels"], mode="eval").body
    if not isinstance(parsed, ast.List):
        raise ValueError("labels_must_be_list")
    records, omitted = [], set()
    for node in parsed.elts:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError("unsupported_constructor")
        try:
            records.extend(parse_guidex_labels(ast.unparse(ast.List(elts=[node], ctx=ast.Load()))))
        except ValueError:
            omitted.add(node.func.id)
    records = [(name, fields) for name, fields in records if name not in omitted]
    # Extract only declared class/field docstrings. Imports are never executed.
    tree = ast.parse(row["text"])
    descriptions = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.bases or node.keywords or any(
                not isinstance(d, ast.Name) or d.id != "dataclass" for d in node.decorator_list
            ):
                raise ValueError("unsupported_class")
            fields = {}
            for index, part in enumerate(node.body):
                if isinstance(part, ast.AnnAssign) and isinstance(part.target, ast.Name):
                    if part.value is not None:
                        raise ValueError("field_default_not_supported")
                    following = node.body[index + 1] if index + 1 < len(node.body) else None
                    desc = following.value.value if (isinstance(following, ast.Expr)
                           and isinstance(following.value, ast.Constant)
                           and isinstance(following.value.value, str)) else part.target.id.replace("_", " ")
                    # Exclude authored examples, which can disclose the row's answers.
                    desc = re.split(r"e\.g\.|[Ee]xamples?:", desc)[0].strip()
                    fields[part.target.id] = desc or part.target.id.replace("_", " ")
                elif not (isinstance(part, ast.Expr) and isinstance(part.value, ast.Constant)
                          and isinstance(part.value.value, str)):
                    raise ValueError("unsupported_class_body")
            descriptions[node.name] = fields
    tasks, gold = [], {}
    for name, fields in records:
        if name not in descriptions or set(fields) != set(descriptions[name]):
            raise ValueError("schema_target_mismatch")
        schema_text = name + " " + " ".join(descriptions[name].values())
        if any(len(value) >= 3 and value in schema_text for value in fields.values()):
            raise ValueError("answer_in_schema_description")
        # No unproven role reassignment: preserve supplied records and mark labels Silver.
        answer = {k: {"status": "present", "items": [unique_span(text, value)]}
                  for k, value in fields.items()}
        if name not in gold:
            tasks.append({"id": name, "kind": "records", "description": name,
                          "fields": descriptions[name], "anchor": next(iter(fields))})
            gold[name] = {"complete": True, "records": []}
        gold[name]["records"].append(answer)
    if not tasks:
        raise ValueError("no_representable_tasks")
    group = "guidex:" + hashlib.sha256(text.encode()).hexdigest()
    return envelope("guidex", group[7:], group, text, "en", "schema_records", "web_text", "silver", tasks, gold)


def import_parquet(path, source, output, limit=100):
    import pyarrow.parquet as pq

    counts, episodes = Counter(), []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100):
        for row in batch.to_pylist():
            if counts["read"] >= limit:
                break
            counts["read"] += 1
            try:
                episode = qa(row, source) if source in {"germanquad", "squad_v2"} else (
                    gretel(row) if source == "gretel" else guidex(row))
                episodes.append(episode)
                counts["accepted"] += 1
            except (ValueError, KeyError, SyntaxError) as exc:
                counts["rejected"] += 1
                counts["reason:" + str(exc).splitlines()[0][:120]] += 1
        if counts["read"] >= limit:
            break
    report = write_corpus(episodes, output)
    report["import"] = dict(counts)
    report["source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    Path(output, "manifest.json").write_text(dumps(report, indent=2))
    return report


def multispan(row):
    tokens, labels = row["context"], row["label"]
    if len(tokens) != len(labels):
        raise ValueError("Token/label length mismatch")
    text, offsets = "", []
    for token in tokens:
        if text:
            text += " "
        offsets.append(len(text))
        text += token
    items, start = [], None
    for i, label in enumerate([*labels, "O"]):
        if label not in {"B", "I", "O"}:
            raise ValueError("Unknown BIO label")
        if start is not None and label in {"B", "O"}:
            end = offsets[i - 1] + len(tokens[i - 1])
            items.append({"start": offsets[start], "end": end, "text": text[offsets[start]:end]})
            start = None
        if label == "B":
            start = i
        elif label == "I" and start is None:
            raise ValueError("Invalid BIO continuation")
    group = "multispan:" + hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()
    return envelope("multispan", row["id"], group, text, "en", "multiple_answers", "tokenized_article",
                    "human", [{"id": "qa", "kind": "entities", "description": "All required answers",
                               "fields": {"answer": " ".join(row["question"])}}],
                    {"qa": {"complete": True, "records": [{"answer": {
                        "status": "present" if items else "absent", "items": items}}]}})


def docred(row):
    """Import occurrence-level entities; cluster relations need a separate proven mapping."""
    text, offsets = "", []
    for sentence in row["sents"]:
        if text:
            text += "\n"
        positions = []
        for index, token in enumerate(sentence):
            if index:
                text += " "
            positions.append(len(text))
            text += token
        offsets.append(positions)
    descriptions = {"ORG": "organization", "PER": "person", "LOC": "location",
                    "TIME": "time or date", "NUM": "number", "MISC": "miscellaneous named entity"}
    fields, gold = {}, {}
    for cluster in row["vertexSet"]:
        for mention in cluster:
            kind = mention["type"]
            if kind not in descriptions:
                raise ValueError("Unmapped entity type")
            sentence, (first, last) = mention["sent_id"], mention["pos"]
            start = offsets[sentence][first]
            end = offsets[sentence][last-1] + len(row["sents"][sentence][last-1])
            fields[kind] = descriptions[kind]
            gold.setdefault(kind, {"status": "present", "items": []})["items"].append(
                {"start": start, "end": end, "text": text[start:end]})
    group = "wikipedia:en:" + row["title"].strip().replace("_", " ").casefold()
    return envelope("docred", row["title"], group, text, "en", "document_entities", "tokenized_article",
                    "human", [{"id": "entities", "kind": "entities", "description": "Annotated mentions",
                               "fields": fields}], {"entities": {"complete": True, "records": [gold]}})


def import_json(path, source, output, limit=100):
    data = loads(Path(path).read_text())
    rows = data["data"] if source == "multispan" else data
    episodes, counts = [], Counter()
    for row in rows[:limit]:
        counts["read"] += 1
        try:
            episodes.append(multispan(row) if source == "multispan" else docred(row))
            counts["accepted"] += 1
        except (ValueError, KeyError, IndexError) as exc:
            counts["rejected:" + str(exc)[:120]] += 1
    report = write_corpus(episodes, output)
    report["import"] = dict(counts)
    if source == "docred":
        report["unsupported"] = "Cluster relations: no validated relation-semantic/mention-pair mapping yet"
    report["source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    Path(output, "manifest.json").write_text(dumps(report, indent=2))
    return report
