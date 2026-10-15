"""Explicit local workflows; heavyweight modules are imported only when selected."""

import argparse
from pathlib import Path

from .jsonio import dumps


def main():
    parser = argparse.ArgumentParser(prog="tractor")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-model")
    p.add_argument("--output", default="models/base")
    p = sub.add_parser("serve")
    p.add_argument("--model", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-pending", type=int, default=1)
    p.add_argument("--wait-timeout", type=float, default=1.0)
    p = sub.add_parser("data")
    actions = p.add_subparsers(dest="action", required=True)
    g = actions.add_parser("generate")
    g.add_argument("--output", default="data/bootstrap")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--sizes", type=int, nargs=3, default=(12000, 1500, 1500))
    g = actions.add_parser("reliability")
    g.add_argument("--output", default="data/synthetic")
    g.add_argument("--seed", type=int, default=20261015)
    g.add_argument("--sizes", type=int, nargs=3, default=(120000, 10000, 20000))
    a = actions.add_parser("audit")
    a.add_argument("--input", required=True)
    a.add_argument("--model", help="Also verify every native target and export native JSONL")
    v = actions.add_parser("validate")
    v.add_argument("--input", required=True)
    v.add_argument("--model", required=True)
    v.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    i = actions.add_parser("import")
    i.add_argument("--input", required=True)
    i.add_argument("--output", required=True)
    p = sub.add_parser("train")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p = sub.add_parser("evaluate")
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/bootstrap")
    p.add_argument("--split", choices=["dev", "test", "authored"], default="dev")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--selection")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p = sub.add_parser("export")
    p.add_argument("--run", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--output", default="models/tractor")
    p = sub.add_parser("bench")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--output", default="reports/latency")
    p.add_argument("--profile", choices=["short", "matrix"], default="short")
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if args.command == "prepare-model":
        from .model import prepare

        prepare(args.output)
    elif args.command == "serve":
        import uvicorn

        from .server import create_app

        if args.max_pending < 0 or args.wait_timeout <= 0:
            parser.error("Invalid admission limits")
        uvicorn.run(
            create_app(
                args.model,
                args.device,
                args.precision,
                max_pending=args.max_pending,
                wait_timeout=args.wait_timeout,
            ),
            host=args.host,
            port=args.port,
            workers=1,
        )
    elif args.command == "data":
        from . import data

        if args.action == "generate":
            print(dumps(data.generate(args.output, args.seed, args.sizes), indent=2))
        elif args.action == "reliability":
            from .corpus import generate

            print(dumps(generate(args.output, args.seed, args.sizes), indent=2))
        elif args.action == "audit":
            from .corpus_audit import audit

            print(dumps(audit(args.input, args.model), indent=2))
        elif args.action == "validate":
            from .model import Extractor

            print(
                dumps(
                    data.validate_dataset(args.input, Extractor(args.model, args.device)), indent=2
                )
            )
        else:
            rows = data.read_rows(args.input)
            data.validate_rows(rows)
            if Path(args.output).exists():
                parser.error("Import destination already exists")
            data.write_rows(args.output, rows)
    elif args.command == "train":
        if args.device == "cpu":
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from .train import train

        print(dumps(train(args.model, args.data, args.config, args.device), indent=2))
    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate(args)
    elif args.command == "export":
        from .evaluate import export

        export(args.run, args.selection, args.output)
    elif args.command == "bench":
        from .bench import bench

        bench(args)


if __name__ == "__main__":
    main()
