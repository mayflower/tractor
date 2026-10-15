"""Measure fresh-process readiness and first HTTP request (disk cache may be warm)."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/tractor")
parser.add_argument("--output", default="reports/cold-start.json")
parser.add_argument("--bench-output")
args = parser.parse_args()
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
started = time.perf_counter()
log = output.with_suffix(".log").open("w")
process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "tractor.cli",
        "serve",
        "--model",
        args.model,
        "--device",
        "cuda",
        "--port",
        str(port),
    ],
    stdout=log,
    stderr=log,
    env={**os.environ, "OMP_NUM_THREADS": "2"},
)
try:
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
        while time.perf_counter() - started < 60:
            if process.poll() is not None:
                raise RuntimeError("Server failed to start; see cold-start.log")
            try:
                if client.get("/readyz").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.05)
        else:
            raise RuntimeError("Readiness timeout")
        ready = time.perf_counter()
        response = client.post(
            "/v1/extract",
            json={
                "text": "Breite 12,5 cm.",
                "fields": {"w": {"type": "quantity", "description": "Breite", "unit": "mm"}},
                "locale": "de-DE",
            },
        )
        finished = time.perf_counter()
        assert response.status_code == 200
        report = {
            "process_start_to_ready_seconds": ready - started,
            "first_request_ms": (finished - ready) * 1000,
            "status": response.status_code,
            "result": response.json(),
            "definition": "Fresh process including model loading, hashing, native warmup; OS file cache may be warm.",
        }
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(report)
    if args.bench_output:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tractor.cli",
                "bench",
                "--url",
                f"http://127.0.0.1:{port}",
                "--output",
                args.bench_output,
                "--count",
                "1000",
                "--warmup",
                "100",
            ],
            check=True,
        )
finally:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log.close()
