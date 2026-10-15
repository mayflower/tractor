"""Reproduce release comparison metrics from captured HTTP responses."""

import argparse
import hashlib
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from .compare import exact, percentile, score_response
from .jsonio import dumps, loads


def rounded_equal(a, b):
    """Same structure and strings, numbers within max(1e-6 absolute, 1e-6 relative)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(rounded_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(rounded_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, (int, Decimal)) and isinstance(b, (int, Decimal)):
        return abs(a - b) <= max(Decimal("0.000001"), abs(b) * Decimal("0.000001"))
    return exact(a, b)


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z / total
    center = (p + z*z / (2*total)) / denominator
    margin = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [center-margin, center+margin]


def summarize(rows, records):
    if len(records) != len(rows) or {r["id"] for r in records} != set(rows):
        raise ValueError("Responses must cover every frozen request exactly once")
    counts = Counter(requests=0, exact=0, data_exact=0, rounded_exact=0,
                     rounded_data_exact=0, errors=0)
    groups = defaultdict(Counter)
    latencies = []
    valid_latencies = []
    for record in records:
        gold = rows[record["id"]]["gold"]["response"]
        result = record.get("result")
        valid = "error" not in record
        scores = score_response(result, gold) if valid else {"exact": False, "data_exact": False}
        delta = dict(requests=1, errors=int(not valid),
                     exact=int(scores["exact"]), data_exact=int(scores["data_exact"]),
                     rounded_exact=int(valid and rounded_equal(result, gold)),
                     rounded_data_exact=int(valid and rounded_equal(result["data"], gold["data"])))
        counts.update(delta)
        groups[rows[record["id"]]["scenario"]].update(delta)
        latencies.append(float(record["latency_ms"]))
        if valid:
            valid_latencies.append(float(record["latency_ms"]))
    return {
        "counts": dict(counts), "groups": dict(groups),
        "rates": {k: counts[k]/counts["requests"] for k in
                  ("exact", "data_exact", "rounded_exact", "rounded_data_exact")},
        "exact_95pct_wilson_interval": wilson(counts["exact"], counts["requests"]),
        "p50_ms": percentile(latencies, .5), "p95_ms": percentile(latencies, .95),
        "valid_response_p50_ms": percentile(valid_latencies, .5),
        "valid_response_p95_ms": percentile(valid_latencies, .95),
        "serial_requests_per_second": len(records) / (sum(latencies)/1000),
        "errors_by_type": dict(Counter(r["error"] for r in records if "error" in r)),
        "http_statuses": dict(Counter(str(r.get("http_status")) for r in records)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="benchmarks/acceptance.jsonl")
    parser.add_argument("--tractor", required=True)
    parser.add_argument("--haiku", required=True)
    parser.add_argument("--luna", required=True)
    parser.add_argument("--output", default="benchmarks/results")
    parser.add_argument("--environment", help="JSON describing the measured hardware")
    args = parser.parse_args()
    data = Path(args.data)
    rows = {r["id"]: r for r in map(loads, data.read_text().splitlines())}
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    environment = loads(Path(args.environment).read_text()) if args.environment else {"unspecified": True}
    report = {"date": None, "requests": len(rows), "data_sha256": data_hash,
              "scope": "synthetic release acceptance subset; prompt-only LLM baselines",
              "numeric_tolerance": {"absolute": "1e-6", "relative_to_gold": "1e-6"},
              "hardware": environment,
              "latency_scope": "HTTP round trip including parsing and scoring; startup excluded; "
                               "tractor loopback, LLMs remote cloud; concurrency one per provider",
              "pricing_per_million_tokens": {"input": .10, "cached_input": .01, "output": .50},
              "pricing_sources": ["https://developers.openai.com/api/docs/pricing",
                                  "https://platform.claude.com/docs/en/about-claude/pricing"],
              "providers": {}}
    order = None
    contract = None
    for name in ("tractor", "haiku", "luna"):
        root = Path(getattr(args, name))
        manifest = loads((root/"manifest.json").read_text())
        if name == "tractor":
            report["date"] = manifest.get("created_at", "").split("T")[0] or None
        if manifest["data_sha256"] != data_hash or manifest["limit"] != len(rows):
            raise ValueError("Manifest does not match frozen data")
        if order is not None and manifest["ids"] != order:
            raise ValueError("Providers received different request orders")
        order = manifest["ids"]
        signature = (manifest["prompt_sha256"], manifest["runner_sha256"])
        if contract is not None and signature != contract:
            raise ValueError("Providers used different prompts or runner code")
        contract = signature
        raw = list(map(loads, (root/"responses.jsonl").read_text().splitlines()))
        records = []
        usage = Counter(input=0, cached_input=0, output=0, cache_write=0)
        models = set()
        for record in raw:
            public = {k: record[k] for k in ("id", "started_at", "latency_ms", "http_status",
                                           "result", "error") if k in record}
            body = record.get("raw_response", {})
            if name == "haiku":
                content = "".join(part.get("text", "") for part in body.get("content", [])
                                  if part.get("type") == "text")
            else:
                content = "".join(part.get("text", "") for item in body.get("output", [])
                                  if item.get("type") == "message"
                                  for part in item.get("content", [])
                                  if part.get("type") == "output_text")
            content = content or record.get("provider_output")
            if content:
                public["provider_output"] = content
            token_usage = body.get("usage", record.get("usage", {}))
            if token_usage:
                public["usage"] = token_usage
                usage["input"] += int(token_usage.get("input_tokens", 0))
                usage["output"] += int(token_usage.get("output_tokens", 0))
                usage["cached_input"] += int(token_usage.get("cache_read_input_tokens", 0))
                usage["cache_write"] += int(token_usage.get("cache_creation_input_tokens", 0))
                if name == "luna":
                    cached = int(token_usage.get("input_tokens_details", {}).get("cached_tokens", 0))
                    usage["input"] -= cached
                    usage["cached_input"] += cached
            model = body.get("model", record.get("model"))
            if model:
                models.add(model)
                public["model"] = model
            records.append(public)
        summary = summarize(rows, records)
        summary.update(manifest=manifest, usage=dict(usage), reported_models=sorted(models),
                       responses_without_usage=sum("usage" not in r for r in records)
                       if name != "tractor" else None)
        summary["estimated_api_cost_usd"] = None if name == "tractor" else (
            usage["input"]*.10 + usage["cached_input"]*.01 + usage["output"]*.50
            + usage["cache_write"]*.125) / 1_000_000
        report["providers"][name] = summary
        destination = output/name
        destination.mkdir(exist_ok=True)
        (destination/"responses.jsonl").write_text("".join(dumps(r)+"\n" for r in records))
        (destination/"manifest.json").write_text(dumps(manifest, indent=2)+"\n")
    (output/"summary.json").write_text(dumps(report, indent=2)+"\n")
    print(dumps({k: {key: val for key, val in v.items() if key not in
                    ("manifest", "groups")} for k, v in report["providers"].items()}, indent=2))


if __name__ == "__main__":
    main()
