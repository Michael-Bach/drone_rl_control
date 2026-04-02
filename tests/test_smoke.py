"""Smoke tests: train.py runs 10 steps for each algorithm without raising."""

import subprocess
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)


def _run(extra_args: list) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/train.py",
         "--config", "configs/test.yaml", "--max-steps", "10",
         *extra_args],
        capture_output=True, text=True, cwd=_ROOT,
    )


def test_td3_single_drone():
    r = _run(["--algo", "td3"])
    assert r.returncode == 0, f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"


def test_sac_single_drone():
    r = _run(["--algo", "sac"])
    assert r.returncode == 0, f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"


def test_ddpg_single_drone():
    r = _run(["--algo", "ddpg"])
    assert r.returncode == 0, f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"


def test_td3_swarm():
    r = _run(["--algo", "td3", "--swarm"])
    assert r.returncode == 0, f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"


def test_sac_swarm():
    r = _run(["--algo", "sac", "--swarm"])
    assert r.returncode == 0, f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"


def test_ddpg_swarm():
    r = _run(["--algo", "ddpg", "--swarm"])
    assert r.returncode == 0, f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"
