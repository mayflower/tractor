"""Reproducible FP32 evaluation, including inside a mixed-precision trainer."""
from contextlib import contextmanager

EVALUATION_PRECISION = {
    "float32_matmul_precision": "highest",
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "inference_mode": True,
}


@contextmanager
def evaluation_precision():
    """Match the production FP32 defaults and restore the caller's settings."""
    import torch

    previous = (torch.get_float32_matmul_precision(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        with torch.inference_mode():
            yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]
