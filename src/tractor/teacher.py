"""Explicit offline Teacher calls with durable reservation limits and response caching."""

import fcntl
import hashlib
import os
import re
from pathlib import Path

import httpx

from .compare import provider_request
from .jsonio import dumps, loads


def reserve(path, limits, input_tokens, output_tokens, short_haiku=False, short_luna=False):
    """Reserve pessimistic usage before each network attempt, including failed attempts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Conservative long-context Haiku 5.5 rates, no caching/batch discounts assumed.
    if short_haiku and short_luna:
        raise ValueError('Select one pricing profile')
    if (short_haiku or short_luna) and input_tokens > 100000:
        raise ValueError('Short Haiku pricing requires bounded input <=100000 tokens')
    input_rate, output_rate = (.10, .50) if short_haiku or short_luna else (.50, 2.50)
    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = loads(path.read_text()) if path.exists() else {
            "requests": 0, "input_tokens_reserved": 0, "output_tokens_reserved": 0,
            "usd_reserved": 0,
        }
        next_state = {"requests": state["requests"] + 1,
                      "input_tokens_reserved": state["input_tokens_reserved"] + input_tokens,
                      "output_tokens_reserved": state["output_tokens_reserved"] + output_tokens,
                      "usd_reserved": float(state["usd_reserved"]) + cost}
        for key, limit in [("requests", "teacher_max_requests"),
                           ("input_tokens_reserved", "teacher_max_input_tokens"),
                           ("output_tokens_reserved", "teacher_max_output_tokens"),
                           ("usd_reserved", "max_external_cost_usd")]:
            if next_state[key] > limits[limit]:
                raise ValueError(f"Teacher budget exhausted: {key}")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(dumps(next_state, indent=2))
        temporary.replace(path)
        return next_state


def settle_success(path, reservation, input_bound, output_bound, usage):
    """Release unused short-context reservation once, using provider-reported usage.

    A durable claim precedes the ledger credit: interruption can over-reserve but
    cannot credit twice. Failed/unknown-usage calls retain their full reservation.
    Both supported short profiles use USD .10/.50 per million; no discounts assumed.
    """
    cache_write = usage.get('cache_creation_input_tokens', 0)
    actual_in = usage['input_tokens'] + cache_write + usage.get('cache_read_input_tokens', 0)
    actual_out = usage['output_tokens']
    if not (0 <= actual_in <= input_bound and 0 <= actual_out <= output_bound):
        raise ValueError('Provider usage exceeds reserved bounds')
    path = Path(path)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = loads(path.read_text())
        state['usd_reserved'] = float(state['usd_reserved'])
        receipts = path.with_suffix('.settled')
        receipts.mkdir(exist_ok=True)
        receipt = receipts / f'{reservation["requests"]}.json'
        if receipt.exists():
            return state
        refund = ((input_bound-actual_in) * .10 + (output_bound-actual_out) * .50 - cache_write * .025) / 1_000_000
        if refund < 0:
            raise ValueError('Usage price exceeds reserved bound')
        # Write the no-repeat claim before crediting the ledger (conservative on crash).
        with receipt.open('x') as stream:
            stream.write(dumps({'request_number': reservation['requests'], 'usage': usage,
                                'reserved_input': input_bound, 'reserved_output': output_bound,
                                'unused_usd': refund}))
            stream.flush()
            os.fsync(stream.fileno())
        state['usd_reserved'] = max(0, state['usd_reserved'] - refund)
        state['input_tokens_reserved'] -= input_bound-actual_in
        state['output_tokens_reserved'] -= output_bound-actual_out
        temporary = path.with_suffix('.tmp')
        temporary.write_text(dumps(state, indent=2))
        temporary.replace(path)
        return state


def parse_teacher(raw):
    if raw.get("stop_reason") != "end_turn":
        raise ValueError("Incomplete Teacher response")
    text = "".join(part["text"] for part in raw.get("content", []) if part.get("type") == "text").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    parsed = loads(fenced[1] if fenced else text)
    if not isinstance(parsed, dict):
        raise ValueError("Teacher response must be a JSON object")
    return parsed


def generate(input_path, prompt_path, output, model, limit=1,
             budget="configs/generalist-budget.json", ledger="reports/training/teacher-budget.json"):
    if model != "claude-haiku-5-5":
        raise ValueError("Configure and verify prices before enabling another Teacher model")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is required for explicit Teacher generation")
    if limit < 1:
        raise ValueError("Teacher request limit must be positive")
    prompt = Path(prompt_path).read_text()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    limits = loads(Path(budget).read_text())
    completed = cached = failed = 0
    with httpx.Client(timeout=120) as client, Path(input_path).open() as stream:
        for line in stream:
            if completed + cached + failed >= limit:
                break
            item = loads(line)
            if item.get("split") != "train" or item.get("text_origin") != "synthetic":
                raise ValueError("This Teacher profile only sends explicitly synthetic training inputs")
            key = hashlib.sha256(dumps({"model": model, "prompt": prompt, "input": item}).encode()).hexdigest()
            path = root / (key + ".json")
            if path.exists():
                cached += 1
                continue
            if path.with_suffix(".rejected").exists():
                failed += 1
                continue
            url, headers, body = provider_request("anthropic", item, prompt, 2048)
            body["model"] = model
            # UTF-8 byte count plus framing allowance bounds text input conservatively.
            input_bound = len(dumps(body).encode()) + 4096
            if input_bound > 100000:
                raise ValueError("Teacher input exceeds bounded profile")
            reserve(ledger, limits, input_bound, 2048)
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            raw = loads(response.text)
            try:
                parsed = parse_teacher(raw)
            except ValueError as exc:
                path.with_suffix(".rejected").write_text(dumps({"input_sha256": key,
                    "model": model, "error": str(exc), "raw": raw}, indent=2))
                failed += 1
                continue
            path.write_text(dumps({"input_sha256": key, "model": model, "usage": raw.get("usage"),
                                   "annotation_status": "silver_unverified", "result": parsed}, indent=2))
            completed += 1
    return {"completed": completed, "cache_hits": cached, "invalid_responses": failed, "ledger": loads(Path(ledger).read_text())
            if Path(ledger).exists() else None, "training_import": "not automatic; validate evidence first"}
