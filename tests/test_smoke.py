"""Smoke test: train.py runs 10 steps without raising."""

import subprocess
import sys
from pathlib import Path


def test_train_single_drone_smoke():
    result = subprocess.run(
        [sys.executable, "scripts/train.py",
         "--config", "configs/test.yaml", "--max-steps", "10"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, (
        f"train.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )


def test_train_swarm_smoke():
    result = subprocess.run(
        [sys.executable, "scripts/train.py",
         "--config", "configs/test.yaml", "--swarm", "--max-steps", "10"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, (
        f"train.py (swarm) failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
