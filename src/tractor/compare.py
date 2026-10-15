"""Frozen-request HTTP comparison; no ML imports, retries or answer repair."""

import argparse
import hashlib
import math
import os
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .jsonio import dumps, loads

MODELS = {"openai": "gpt-6-luna", "anthropic": "claude-haiku-5-5"}


def select_rows(path, limit):
    """Stratified, hash-ordered selection independent of gold and predictions."""
    if limit < 1:
        raise ValueError("limit must be positive")
    groups = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        row = loads(line)
        groups[(row.get("scenario", "authored"), row.get("metadata", {}).get("language", ""))].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    selected = []
    index = 0
    while len(selected) < limit:
        added = False
        for key in sorted(groups):
            if index < len(groups[key]):
                selected.append(groups[key][index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        index += 1
    if len(selected) != limit:
        raise ValueError("Requested more examples than available")
    return selected


def exact(a, b):
    """Exact JSON numbers (Decimal), with booleans distinct from numbers."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def score_response(result, gold):
    if not isinstance(result, dict) or set(result) != {"data", "issues"}:
        raise ValueError("Response must contain exactly data and issues")
    if not isinstance(result["issues"], list):
        raise ValueError("issues must be an array")
    return {"exact": exact(result, gold), "data_exact": exact(result["data"], gold["data"])}


def provider_request(provider, request, prompt, max_tokens):
    user = dumps({k: v for k, v in request.items() if k != "include_details"})
    if provider == "openai":
        return "https://api.openai.com/v1/responses", {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        }, {
            "model": MODELS[provider], "instructions": prompt, "input": user,
            "reasoning": {"effort": "none"}, "max_output_tokens": max_tokens,
            "store": False,
        }
    headers = {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
    }
    if os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        headers["anthropic-workspace-id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
    return "https://api.anthropic.com/v1/messages", headers, {
        "model": MODELS[provider], "system": prompt,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens, "thinking": {"type": "disabled"},
    }


def parse_provider(provider, body):
    if provider == "openai":
        if body.get("status") != "completed":
            raise ValueError("Incomplete provider response")
        text = "".join(
            part["text"] for item in body.get("output", [])
            if item.get("type") == "message" for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
    else:
        if body.get("stop_reason") != "end_turn":
            raise ValueError("Incomplete provider response")
        text = "".join(p["text"] for p in body.get("content", []) if p.get("type") == "text")
    return loads(text)


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["tractor", *MODELS], required=True)
    parser.add_argument("--data", default="data/synthetic/test.jsonl")
    parser.add_argument("--limit", type=int, default=680)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="benchmarks/prompt.txt")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-sha256", help="Required for local model provenance")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    rows = select_rows(args.data, args.limit)
    prompt = Path(args.prompt).read_text()
    if args.max_output_tokens < 1:
        parser.error("max-output-tokens must be positive")
    if not args.dry_run:
        if args.provider == "tractor" and (
            not args.model_sha256 or len(args.model_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.model_sha256)
        ):
            parser.error("tractor requires --model-sha256 (64 lowercase hex digits)")
        key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(args.provider)
        if key and not os.environ.get(key):
            parser.error(f"Set {key} in the environment; never put it in CLI arguments")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(), "provider": args.provider,
        "model": MODELS.get(args.provider, "tractor"), "model_sha256": args.model_sha256,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ids": [r["id"] for r in rows], "limit": args.limit,
        "selection": "scenario/language round robin, SHA256(id) order v1",
        "max_output_tokens": args.max_output_tokens, "concurrency": 1, "retries": 0,
        "warmups": 0, "timeout_seconds": 120, "dry_run": args.dry_run,
        "reasoning": "disabled for both API providers", "structured_output": "prompt-only",
        "latency_scope": "client HTTP round trip including parsing; model startup excluded",
    }
    (destination / "manifest.json").write_text(dumps(manifest, indent=2) + "\n")
    (destination / "prompt.txt").write_text(prompt)
    if args.dry_run:
        print(f"Frozen {len(rows)} IDs; no API requests made")
        return
    counts = Counter()
    groups = defaultdict(Counter)
    times = []
    with httpx.Client(timeout=120) as client, (destination / "responses.jsonl").open("w") as raw:
        for row in rows:
            started = time.perf_counter()
            record = {
                "id": row["id"], "exact": False, "data_exact": False,
                "started_at": datetime.now(UTC).isoformat(),
            }
            try:
                if args.provider == "tractor":
                    response = client.post(args.url.rstrip("/") + "/v1/extract", json={
                        **row["request"], "include_details": False,
                    })
                else:
                    url, headers, body = provider_request(
                        args.provider, row["request"], prompt, args.max_output_tokens,
                    )
                    response = client.post(url, headers=headers, json=body)
                record["http_status"] = response.status_code
                record["request_id"] = response.headers.get(
                    "x-request-id", response.headers.get("request-id")
                )
                response.raise_for_status()
                body = loads(response.text)
                record["raw_response"] = body
                result = body if args.provider == "tractor" else parse_provider(args.provider, body)
                record.update(score_response(result, row["gold"]["response"]))
                record["result"] = result
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                # Never serialize exceptions/headers: they can contain credentials or request data.
                record["error"] = type(exc).__name__
            record["latency_ms"] = (time.perf_counter() - started) * 1000
            times.append(record["latency_ms"])
            delta = Counter(requests=1, exact=int(record["exact"]),
                            data_exact=int(record["data_exact"]), errors=int("error" in record))
            counts.update(delta)
            groups[row.get("scenario", "authored")].update(delta)
            raw.write(dumps(record) + "\n")
            raw.flush()
    summary = {
        "counts": dict(counts), "groups": {k: dict(v) for k, v in groups.items()},
        "exact_request_rate": counts["exact"] / counts["requests"],
        "exact_data_rate": counts["data_exact"] / counts["requests"],
        "p50_ms": percentile(times, .5), "p95_ms": percentile(times, .95),
        "p99_ms": percentile(times, .99), "latency_includes_errors": True,
        "p99_reliable_sample": len(times) >= 1000,
        "cost_usd": None, "cost_note": "Usage retained per response; cost not calculated",
    }
    (destination / "summary.json").write_text(dumps(summary, indent=2) + "\n")
    print(dumps(summary, indent=2))


if __name__ == "__main__":
    main()
