# Drone RL Lab — Multi-Algorithm Extension Design Spec

*Date: 2026-04-01*

---

## Overview

Extend the existing single-algorithm TD3 system into a pluggable RL algorithm lab supporting TD3, SAC, and DDPG — all off-policy, all sharing the same replay buffer and environment API. Algorithm selection via YAML config field `agent.type` or CLI flag `--algo` (CLI overrides config). W&B + CSV experiment tracking added alongside.

Radar observation extended: `DroneEnv` gains a `radar_obs` attribute (`float32[4]` = `[x, y, z, t]`), appended to the state vector, raising `OBS_DIM` from 7 to 11.

---

## Architecture

```
src/drone_rl/agents/
  base.py          # AgentBase ABC: select_action, train_step, save, load
  td3.py           # TD3Agent(AgentBase) — refactored, behavior unchanged
  sac.py           # SACAgent(AgentBase) — new
  ddpg.py          # DDPGAgent(AgentBase) — new
  factory.py       # make_agent(cfg, obs_dim, action_dim) -> AgentBase
  replay_buffer.py # unchanged
  noise.py         # unchanged

src/drone_rl/utils/
  logger.py        # TrainingLogger: W&B + CSV

src/drone_rl/envs/
  drone_env.py     # radar_obs attribute added, OBS_DIM 7 → 11

scripts/train.py   # --algo flag + --wandb flag; algorithm-agnostic loop
configs/
  single_drone.yaml / swarm.yaml / test.yaml  # gain agent.type + SAC keys
```

The training loop in `train.py` only calls `AgentBase` methods — it has no knowledge of which algorithm is running.

---

## AgentBase Interface

```python
class AgentBase(ABC):
    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray: ...

    @abstractmethod
    def train_step(self) -> dict: ...  # returns metric dict; keys vary by algorithm

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
```

`RunningMeanStd` (Welford normalization) and `ReplayBuffer` are shared utilities imported by each algorithm — not in the base class.

---

## Algorithms

All three algorithms share: twin-critic or single-critic architecture on MLP networks, `ReplayBuffer`, `RunningMeanStd` for optional obs/reward normalization.

### DDPG
- Single critic, single target pair (actor + critic)
- OUNoise exploration from existing `noise.py`
- Softest update rule (Polyak τ)
- Baseline: simplest algorithm, establishes lower bound on performance

### TD3 (refactored)
- Twin critics (clipped double Q-learning)
- Delayed actor updates (`policy_delay` steps)
- Target policy smoothing with clipped Gaussian noise
- Behavior identical to current implementation; only inheritance changes

### SAC
- Twin critics
- Actor outputs `(mean, log_std)` → reparameterization trick → squashed Gaussian (tanh)
- No external exploration noise — entropy built into policy
- Auto-tunes temperature α against target entropy `-action_dim`
- Extra parameters: `lr_alpha`, `init_alpha`, `auto_tune_alpha`

| | DDPG | TD3 | SAC |
|---|---|---|---|
| Critics | 1 | 2 | 2 |
| Actor updates | every step | every `policy_delay` | every step |
| Exploration | OUNoise | Gaussian + smoothing | entropy |
| Extra param | — | `policy_delay` | `lr_alpha`, `init_alpha` |

---

## Radar Observation

`DroneEnv` gains:
```python
self.radar_obs: np.ndarray  # shape (4,), dtype float32, default [0, 0, 0, 0]
```

`_obs()` appends it: observation = `[x, y, z, vx, vy, vz, yaw, radar_x, radar_y, radar_z, radar_t]` → `OBS_DIM = 11`.

**No change to `step()` signature** — Gymnasium API intact. Callers set the attribute before calling `step()`:
- **Simulation (DroneEnv standalone):** stays `[0, 0, 0, 0]` by default
- **TelloEnv:** reads `[x, y, z, t]` from radar station in `_read_telemetry()`, sets `self.radar_obs`
- **SwarmEnv:** each drone's `radar_obs` set independently — drones may receive different radar readings

`observation_space` updated to `Box(-inf, inf, shape=(11,), dtype=float32)`.

---

## Experiment Tracking

`TrainingLogger` in `src/drone_rl/utils/logger.py`:

- **W&B:** initialized if `cfg["training"].get("wandb", False)` or `--wandb` CLI flag. Missing API key → silent skip (no crash).
- **CSV:** always written to `outputs/<run_name>/train_log.csv`. One row per episode.

Metrics logged each episode:

| Metric | Source |
|---|---|
| `episode_reward` | accumulated step rewards |
| `episode_length` | step count |
| `coverage_fraction` | `reward_fn.coverage_fraction` |
| `critic_loss` | `train_step()` dict |
| `actor_loss` | `train_step()` dict |
| `alpha` | SAC only (entropy coefficient) |
| `steps_per_second` | wall-clock timing |

`train_step()` returns whatever keys it has — logger writes all present keys. No logger changes needed when adding metrics to an algorithm.

---

## Config Schema

`agent.type` added to all YAML configs. Algorithm-specific keys are ignored by other algorithms.

```yaml
agent:
  type: sac                 # td3 | sac | ddpg  (overridable via --algo)
  hidden_dims: [256, 256]
  lr_actor: 3.0e-4
  lr_critic: 3.0e-4
  gamma: 0.99
  tau: 0.005
  batch_size: 256
  buffer_size: 1_000_000
  normalize_obs: true
  normalize_rew: true
  # td3 / ddpg only
  policy_delay: 2
  noise:
    type: gaussian
    sigma: 0.1
  # sac only
  lr_alpha: 3.0e-4
  init_alpha: 0.2
  auto_tune_alpha: true
```

CLI:
```bash
python scripts/train.py --config configs/single_drone.yaml --algo sac --wandb
```

`--algo` overrides `cfg["agent"]["type"]`. `--wandb` overrides `cfg["training"]["wandb"]`. Both applied before any config key is read downstream.

`make_agent(cfg, obs_dim, action_dim)` reads `cfg["agent"]["type"]`, instantiates the right class. Unknown type → `ValueError` listing valid options.

---

## Testing

Existing 47 tests remain green. New tests:

| File | Covers |
|---|---|
| `tests/test_agents.py` | All three agents: `select_action` output shape, `train_step` returns dict with expected keys, `save`/`load` round-trip |
| `tests/test_factory.py` | `make_agent` routes correctly for all three type strings; `ValueError` on unknown type |
| `tests/test_radar_obs.py` | `DroneEnv` obs shape `(11,)`; default radar zeros; setting `radar_obs` changes returned obs |
| `tests/test_logger.py` | CSV written with correct columns; W&B skipped when not configured |
| `tests/test_smoke.py` | Extended: one smoke run per algo (`td3`, `sac`, `ddpg`) using `test.yaml` |

All tests use `test.yaml` (tiny env, 2 drones, 20-step episodes). No W&B credentials required in CI.

---

## Tech Stack Changes

| Addition | Library |
|---|---|
| Experiment tracking | `wandb>=0.16` (already in naval project's optional deps) |
| No new deps otherwise | SAC/DDPG built on existing `torch`, `numpy` |
