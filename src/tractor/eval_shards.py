"""Run existing native evaluation in independent processes; merge integer counters."""

import argparse
import asyncio
import hashlib
import sys
from collections import Counter
from pathlib import Path

from .evaluate import metrics
from .jsonio import dumps, loads


def merge(parts, expected_requests):
    combined = {}
    for summaries in parts:
        keys = set()
        for summary in summaries:
            key = (summary["model"], tuple(sorted(summary["thresholds"].items())))
            if key in keys:
                raise ValueError("Duplicate threshold result in shard")
            keys.add(key)
            entry = combined.setdefault(
                key,
                {
                    "template": summary,
                    "counts": Counter(),
                    "groups": {},
                    "normalizer": Counter(),
                    "predictions": [],
                    "shards": 0,
                },
            )
            entry["counts"].update(summary["metrics"]["counts"])
            entry["normalizer"].update(summary["normalizer_on_gold"])
            for group, scores in summary["groups"].items():
                entry["groups"].setdefault(group, Counter()).update(scores["counts"])
            entry["predictions"].append(summary["predictions"])
            entry["shards"] += 1
    result = []
    for entry in combined.values():
        if entry["shards"] != len(parts) or entry["counts"]["requests"] != expected_requests:
            raise ValueError("Incomplete shard evaluation")
        result.append(
            entry["template"]
            | {
                "metrics": metrics(entry["counts"]),
                "groups": {k: metrics(v) for k, v in entry["groups"].items()},
                "normalizer_on_gold": dict(entry["normalizer"]),
                "predictions": entry["predictions"],
                "shards": entry["shards"],
            }
        )
    if not result:
        raise ValueError("No shard results")
    return result


async def run(args):
    if args.workers < 1 or (args.calibrate and args.split != "dev"):
        raise ValueError("Positive workers and dev-only calibration required")
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Evaluation output must be empty")
    if not Path(args.model, "config.json").exists():
        raise ValueError("Supply one immutable checkpoint/bundle, not a run directory")
    source = Path(args.data) / f"{args.split}.jsonl"
    data = source.read_bytes()
    lines = data.splitlines(keepends=True)
    ids = [loads(line)["id"] for line in lines]
    if len(set(ids)) != len(ids) or len(lines) < args.workers:
        raise ValueError("Duplicate IDs or too few rows")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(source.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "requests": len(lines),
        "workers": args.workers,
        "split": args.split,
        "calibrate": args.calibrate,
    }
    (root / "shards.json").write_text(dumps(manifest, indent=2))
    processes, logs, destinations = [], [], []
    try:
        for i in range(args.workers):
            shard = root / "shards" / str(i)
            dataset, output = shard / "data", shard / "results"
            dataset.mkdir(parents=True)
            start, end = len(lines) * i // args.workers, len(lines) * (i + 1) // args.workers
            (dataset / f"{args.split}.jsonl").write_bytes(b"".join(lines[start:end]))
            command = [
                sys.executable,
                "-m",
                "tractor.cli",
                "evaluate",
                "--model",
                args.model,
                "--data",
                str(dataset),
                "--split",
                args.split,
                "--output",
                str(output),
                "--device",
                args.device,
            ]
            if args.calibrate:
                command.append("--calibrate")
            if args.selection:
                command.extend(["--selection", args.selection])
            log = (shard / "worker.log").open("w")
            logs.append(log)
            processes.append(
                await asyncio.create_subprocess_exec(
                    *command, stdout=log, stderr=asyncio.subprocess.STDOUT
                )
            )
            destinations.append(output)
        statuses = await asyncio.gather(*(p.wait() for p in processes))
        if any(statuses):
            raise RuntimeError(f"Evaluation workers failed: {statuses}; inspect worker.log")
        merged = merge([loads((d / "summary.json").read_text()) for d in destinations], len(lines))
        (root / "summary.json").write_text(dumps(merged, indent=2))
        best = max(
            merged,
            key=lambda s: (
                s["metrics"]["field_f1"],
                s["metrics"]["exact_request_rate"],
                s["metrics"]["precision"],
            ),
        )
        if args.calibrate:
            (root / "selection.json").write_text(dumps(best, indent=2))
        print(dumps({"metrics": best["metrics"], "thresholds": best["thresholds"]}, indent=2))
    finally:
        for process in processes:
            if process.returncode is None:
                process.terminate()
        await asyncio.gather(*(p.wait() for p in processes))
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--selection")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
