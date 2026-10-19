"""Real native task, target, empty-record and encoder-count integration checks."""

import argparse
from pathlib import Path

from tractor.episodes import Gold
from tractor.generalist_data import generate, validate_native
from tractor.jsonio import dumps
from tractor.model import Extractor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default="models/tractor")
parser.add_argument("--output", default="reports/training/native-integration.json")
args = parser.parse_args()
extractor = Extractor(args.model, "cpu")
samples = {}
for episode in generate(128, 42):
    samples.setdefault(episode.tasks[0].kind, episode)
assert set(samples) == {"entities", "records", "relations", "classification"}
empty = samples["records"].model_copy(update={
    "text": "There are no employees in this document.",
    "gold": {"employment": Gold(complete=True, records=[])},
})
samples["empty_records"] = empty
report = {}
for name, episode in samples.items():
    _, tokens = validate_native(episode, extractor)
    before = extractor.encoder_calls
    output = extractor.extract_tasks(episode.text, episode.tasks)
    assert extractor.encoder_calls - before == 1
    if name == "empty_records":
        assert not output.get("employment")
    report[name] = {"tokens": tokens, "targets_verified": True, "encoder_calls": 1,
                    "prediction": output}
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(dumps(report, indent=2))
print("Real native integration passed:", ", ".join(report))
