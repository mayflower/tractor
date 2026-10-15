"""HTTP latency measurement with both closed and scheduled arrival profiles."""

import asyncio
import math
import platform
import time
from collections import Counter
from pathlib import Path

import httpx

from .jsonio import dumps


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)]


def payload(index, fields=4, words=0, fresh=False):
    facts = " ".join(f"Objekt {j} gehört zum Lager." for j in range(words // 6))
    text = (
        f"Produkt Atlas-{index} hat Breite {10 + index % 40} cm, Höhe 20 cm, Gewicht 2 kg und Dauer 3 h. "
        + facts
    )
    specs = [("Breite", "mm"), ("Höhe", "cm"), ("Gewicht", "g"), ("Dauer", "min")]
    mapping = {}
    for i in range(fields):
        label, unit = specs[i % 4]
        mapping[f"f{i}" if not fresh else f"f{index}_{i}"] = {
            "type": "quantity",
            "description": label,
            "unit": unit,
        }
    return {"text": text, "fields": mapping, "locale": "de-DE"}


async def run_profile(client, count, warmup, fields, words, concurrency, rate, fresh):
    for i in range(warmup):
        response = await client.post("/v1/extract", json=payload(i, fields, words, fresh))
        if response.status_code == 422:
            return [], {
                "count": 0,
                "warmup": i + 1,
                "fields": fields,
                "extra_words": words,
                "concurrency": concurrency,
                "offered_rps": rate,
                "fresh_schema": fresh,
                "unsupported": response.json(),
                "statuses": {"422": 1},
            }
        if response.status_code != 200:
            raise RuntimeError(f"Warmup failed: {response.status_code}")
    started = time.perf_counter()
    rows = []
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index):
        scheduled = started + index / rate if rate else None
        if scheduled:
            await asyncio.sleep(max(0, scheduled - time.perf_counter()))
        async with semaphore:
            sent = time.perf_counter()
            body = dumps(payload(index, fields, words, fresh)).encode()
            server_timing = input_tokens = None
            try:
                response = await client.post(
                    "/v1/extract", content=body, headers={"Content-Type": "application/json"}
                )
                status = response.status_code
                server_timing = response.headers.get("server-timing")
                input_tokens = response.headers.get("x-tractor-input-tokens")
            except httpx.TimeoutException:
                status = "timeout"
            except httpx.TransportError:
                status = "transport_error"
            end = time.perf_counter()
            rows.append(
                {
                    "index": index,
                    "status": status,
                    "latency_ms": (end - sent) * 1000,
                    "scheduled_latency_ms": (end - (scheduled or sent)) * 1000,
                    "send_delay_ms": (sent - (scheduled or sent)) * 1000,
                    "request_bytes": len(body),
                    "server_timing": server_timing,
                    "input_tokens": input_tokens,
                    "fields": fields,
                    "extra_words": words,
                }
            )

    if rate:
        # Scheduled arrivals; queued client tasks remain included in observed latency.
        await asyncio.gather(*(one(i) for i in range(count)))
    else:
        cursor = iter(range(count))

        async def worker():
            for i in cursor:
                await one(i)

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - started
    statuses = Counter(str(r["status"]) for r in rows)
    good = [r["latency_ms"] for r in rows if r["status"] == 200]
    all_latencies = [r["scheduled_latency_ms"] for r in rows]
    return rows, {
        "count": count,
        "warmup": warmup,
        "fields": fields,
        "extra_words": words,
        "concurrency": concurrency,
        "offered_rps": rate,
        "fresh_schema": fresh,
        "elapsed_seconds": elapsed,
        "successful_rps": len(good) / elapsed,
        "statuses": dict(statuses),
        "success_fraction": len(good) / count,
        "rejection_fraction": statuses["503"] / count,
        "timeout_fraction": statuses["timeout"] / count,
        "p50_ms": percentile(good, 0.5),
        "p95_ms": percentile(good, 0.95),
        "p99_ms": percentile(good, 0.99),
        "scheduled_p95_ms": percentile(all_latencies, 0.95),
        "p99_reliable_sample": len(good) >= 1000,
    }


async def _bench(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    profiles = [(4, 0, 1, None, False)]
    if args.profile == "matrix":
        profiles += [(n, 0, 1, None, False) for n in (1, 8, 16, 32)]
        profiles += [(4, n, 1, None, False) for n in (60, 120, 180)]
        profiles += [(4, 0, n, None, False) for n in (2, 4, 8)]
        profiles += [(4, 0, 64, rate, False) for rate in (10, 25, 50)]
        profiles += [(4, 0, 1, None, True)]
    summaries = []
    async with httpx.AsyncClient(
        base_url=args.url, timeout=10.0, limits=httpx.Limits(max_connections=64)
    ) as client:
        for i, (fields, words, concurrency, rate, fresh) in enumerate(profiles):
            rows, summary = await run_profile(
                client, args.count, args.warmup, fields, words, concurrency, rate, fresh
            )
            summaries.append(summary)
            Path(root, f"profile-{i}.jsonl").write_text("".join(dumps(row) + "\n" for row in rows))
            Path(root, "summary.json").write_text(
                dumps(
                    {
                        "client_platform": platform.platform(),
                        "url": args.url,
                        "profiles": summaries,
                    },
                    indent=2,
                )
            )
            print(dumps(summary), flush=True)


def bench(args):
    if args.count < 1 or args.warmup < 0:
        raise ValueError("Invalid benchmark counts")
    asyncio.run(_bench(args))
