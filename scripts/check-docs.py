"""Check manual links, snippet syntax, and request fixtures without model inference."""

import ast
import re
import subprocess
from pathlib import Path

from tractor.jsonio import loads
from tractor.types import ExtractRequest

root = Path(__file__).resolve().parents[1]
pages = list((root/"docs/manual").rglob("*.md")) + [
    root/"README.md", root/"docs/api.md", root/"release/MODEL_CARD.md",
    root/"examples/manual/README.md",
]
snippets = 0
for path in pages:
    text = path.read_text()
    for link in re.findall(r"\]\(([^)]+)\)", text):
        if link.startswith(("https://", "http://", "#")):
            continue
        target = link.split("#", 1)[0]
        assert (path.parent/target).exists(), f"Broken link in {path}: {link}"
    for language, code in re.findall(r"```(\w+)\n(.*?)```", text, flags=re.S):
        if language == "python":
            ast.parse(code, filename=str(path))
        elif language == "json":
            parsed = loads(code)
            if isinstance(parsed, dict) and "text" in parsed and "fields" in parsed:
                ExtractRequest.model_validate(parsed)
        elif language == "jsonl":
            for line in code.splitlines():
                if line.strip():
                    loads(line)
        elif language == "sh":
            subprocess.run(["bash", "-n"], input=code, text=True, check=True,
                           capture_output=True)
        snippets += 1
for name in ("scalar.json", "records.json"):
    ExtractRequest.model_validate(loads((root/"examples/manual"/name).read_text()))
cases = (root/"examples/manual/cases.jsonl").read_text().splitlines()
for line in cases:
    ExtractRequest.model_validate(loads(line)["request"])
print(f"Checked {len(pages)} documentation pages, {snippets} snippets, and {len(cases)} cases.")
