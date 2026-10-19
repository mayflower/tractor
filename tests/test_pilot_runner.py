import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).parents[1] / 'scripts' / 'run-generalist-pilot.py'
    spec = importlib.util.spec_from_file_location('pilot_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, 'sleep', lambda _seconds: None)
    return module


def test_busy_shared_gpu_is_not_used(runner, monkeypatch):
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='35000, 100\n'))
    with pytest.raises(RuntimeError, match='no GPU work started'):
        runner.require_idle_gpu()


def test_recent_utilization_can_settle_before_next_arm(runner, monkeypatch):
    samples = iter(['344, 80\n', '344, 0\n', '344, 0\n', '344, 0\n'])
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=next(samples)))
    runner.require_idle_gpu()
