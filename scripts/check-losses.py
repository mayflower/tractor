"""Check real boundary and record losses/gradients without an optimizer update."""

from pathlib import Path

import torch
from gliner2.training.trainer import ExtractorCollator

from tractor.data import read_rows
from tractor.jsonio import dumps
from tractor.model import Extractor, native_example
from tractor.training_adapter import configure_processor

Path("reports").mkdir(exist_ok=True)
x = Extractor("models/tractor", "cpu")
configure_processor(x.model.processor)
rows = read_rows("data/bootstrap/train.jsonl")
rows = [rows[0], rows[2]]
collator = ExtractorCollator(
    x.model.processor, is_training=True, architecture="boundary", max_gold_per_query=64
)
native = [native_example(r) for r in rows]
batch = collator([(n["input"], n["output"]) for n in native])
x.model.train()
output = x.model(batch)
print(type(output), output.losses)
losses = {
    k: float(v.detach()) for k, v in output.losses.items() if torch.is_tensor(v) and v.numel() == 1
}
assert losses["record_field_loss"] > 0 and losses["pair_loss"] > 0
assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
output.total_loss.backward()
gradients = {
    name: float(param.grad.abs().sum())
    for name, param in x.model.named_parameters()
    if param.grad is not None
    and (name.startswith("record_decoder.") or name.startswith("boundary_head."))
}
assert any(v > 0 for name, v in gradients.items() if name.startswith("record_decoder."))
assert any(v > 0 for name, v in gradients.items() if name.startswith("boundary_head."))
Path("reports/native-loss-check.json").write_text(
    dumps({"losses": losses, "gradient_l1": gradients}, indent=2)
)
print("Actual entity/boundary and record loss/gradient paths verified.")
