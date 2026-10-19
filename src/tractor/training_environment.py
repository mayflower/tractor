"""Read-only local training inventory; no downloads or GPU benchmarks."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path


def inspect_environment(models, output, load=False):
    import torch

    from .model import COMMIT, REVISION

    report = {
        "python": platform.python_version(),
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "gliner2", "transformers")},
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
        "gliner_commit": COMMIT,
        "base_revision": REVISION,
        "disk_free_bytes": shutil.disk_usage(".").free,
        "cuda_available": torch.cuda.is_available(),
        "models": [],
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report["gpu"] = {
            "name": torch.cuda.get_device_name(), "free_bytes": free, "total_bytes": total,
            "bf16": torch.cuda.is_bf16_supported(),
        }
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        report["gpu"]["utilization_percent_memory_mib"] = result.stdout.strip()
    for value in models:
        path = Path(value)
        item = {"path": str(path), "exists": path.joinpath("model.safetensors").is_file()}
        if item["exists"]:
            with path.joinpath("model.safetensors").open("rb") as stream:
                item["weights_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            config = json.loads(path.joinpath("config.json").read_text())
            item["config"] = {k: config.get(k) for k in ("architecture", "max_len", "boundary")}
            if load:
                from .model import Extractor

                extractor = Extractor(path, "cpu")
                item["loaded"] = True
                item["joint_token_limit"] = extractor.input_limit
                item["record_capacity"] = extractor.model.boundary_settings.record_instance_queries
                item["field_capacity"] = extractor.model.boundary_settings.max_gold_per_query
                del extractor
        report["models"].append(item)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report
