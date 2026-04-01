# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode with dev deps
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test
pytest tests/test_drone_env.py::test_obs_shape -v

# Run smoke test only
pytest tests/test_smoke.py -v

# Train single drone
python scripts/train.py --config configs/single_drone.yaml

# Train swarm
python scripts/train.py --config configs/swarm.yaml --swarm

# Evaluate a checkpoint
python scripts/evaluate.py --config configs/single_drone.yaml --checkpoint outputs/checkpoint_1000.pt

# Format
black src/ tests/ scripts/

# Lint
flake8 src/ tests/ scripts/ --max-line-length 100
```

## Architecture

Two-phase system built around layered Gymnasium environments:

**Phase 1 (single drone):** `DroneEnv` — point-mass physics matching DJI Tello limits (max speed 8 m/s, max altitude 10 m). State: `[x, y, z, vx, vy, vz, yaw]`. Action: `[thrust, roll, pitch, yaw_rate]` ∈ [-1, 1].

**Phase 2 (swarm):** `SwarmEnv` wraps N `DroneEnv` instances and presents a single flat `(N×7,)` observation and `(N×4,)` action to the centralized TD3 controller. All drones share a single `CoverageReward` instance so overlapping coverage generates no bonus.

**Reward:** `CoverageReward` discretizes the patrol area into a grid. Newly visited cells yield `+coverage_bonus`, previously visited cells yield `-revisit_penalty`, boundary violations yield `-boundary_penalty`. Swarm phase adds `-proximity_penalty` for drones closer than `min_separation`.

**TD3 agent:** Lives in `src/drone_rl/agents/td3.py`. Uses twin critics, delayed actor updates, target policy smoothing, optional obs/reward normalisation. Same implementation pattern as the sibling `adversarial-reinforcement-learning-naval-warfare` project.

**Sim-to-real:** `TelloEnv` in `src/drone_rl/utils/hardware.py` implements the same `reset()` / `step()` interface as `DroneEnv` but calls `djitellopy` SDK instead of simulating physics. Swap `DroneEnv` for `TelloEnv` at deploy time — `SwarmEnv` and TD3 are untouched.

## Key Conventions

- All configs are YAML; load with `yaml.safe_load`. The `cfg["env"]`, `cfg["reward"]`, `cfg["agent"]`, `cfg["training"]` keys are always present.
- Noise is constructed via `make_noise(cfg["agent"]["noise"])` factory in `agents/noise.py`.
- Checkpoints save actor, both critics, and obs normalisation stats via `agent.save(path)`.
- `DroneEnv` accepts an optional `reward_fn: CoverageReward` argument. Pass `None` for standalone use (creates its own); pass the shared instance from `SwarmEnv` for swarm use. When `reward_fn` is passed externally, `DroneEnv` does NOT call `reward_fn.reset()` on episode reset — the caller owns that.
