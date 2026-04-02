# drone-rl-control

Reinforcement learning lab for autonomous drone patrol and swarm coverage. Built around a Gymnasium-compatible point-mass simulator calibrated to DJI Tello limits, with pluggable off-policy algorithms and optional real-hardware deployment.

## Algorithms

| Algorithm | Key traits |
|-----------|-----------|
| **TD3** | Twin critics, delayed actor updates, target policy smoothing |
| **SAC** | Squashed Gaussian actor, auto-entropy tuning, no external noise |
| **DDPG** | Single critic, OU/Gaussian exploration noise, every-step updates |

All three share the same `AgentBase` interface and are swappable via a single flag.

## Environments

**`DroneEnv`** — single point-mass drone in a bounded 3-D area.

| | |
|---|---|
| Observation | `[x, y, z, vx, vy, vz, yaw, radar_x, radar_y, radar_z, radar_t]` — 11-dim float32 |
| Action | `[thrust, roll, pitch, yaw_rate]` ∈ [-1, 1] — 4-dim float32 |
| Physics | Max speed 8 m/s, max altitude 10 m, max accel 4 m/s², max yaw rate 90°/s |

`radar_obs` (`float32[4]`) is written externally each step from a radar station feed. It is zeroed on `reset()`.

**`SwarmEnv`** — wraps N `DroneEnv` instances behind a single flat observation `(N×11,)` and action `(N×4,)`. All drones share one `CoverageReward` so overlapping coverage yields no bonus. A proximity penalty discourages collision.

## Reward

`CoverageReward` discretises the patrol area into a grid:

| Event | Signal |
|-------|--------|
| New cell visited | `+coverage_bonus` |
| Already-visited cell | `-revisit_penalty` |
| Boundary / altitude violation | `-boundary_penalty` |
| Drone pair closer than `min_separation` | `-proximity_penalty` (swarm only) |

## Installation

```bash
# Core
pip install -e .

# With dev tools (pytest, black, flake8)
pip install -e ".[dev]"

# With W&B experiment tracking
pip install -e ".[wandb]"

# With real Tello hardware support
pip install -e ".[hardware]"
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0.

## Quick start

```bash
# Train single drone with TD3 (default)
python scripts/train.py --config configs/single_drone.yaml

# Train with SAC
python scripts/train.py --config configs/single_drone.yaml --algo sac

# Train swarm with DDPG
python scripts/train.py --config configs/swarm.yaml --swarm --algo ddpg

# Enable W&B logging
python scripts/train.py --config configs/single_drone.yaml --wandb

# Evaluate a checkpoint
python scripts/evaluate.py --config configs/single_drone.yaml \
    --checkpoint outputs/checkpoint_final.pt
```

Algorithm can also be set in the config under `agent.type`; the `--algo` flag overrides it.

## Configuration

All configs live in `configs/`. The four top-level keys are always present:

```yaml
env:
  x_max: 25.0          # patrol area half-extent (m)
  y_max: 25.0
  max_altitude: 10.0
  max_speed: 8.0
  dt: 0.05             # simulation timestep (s)
  max_steps: 500       # episode length

reward:
  cell_size: 2.5       # grid resolution (m)
  coverage_bonus: 1.0
  revisit_penalty: 0.0
  boundary_penalty: 10.0

agent:
  type: td3            # td3 | sac | ddpg
  lr_actor: 0.0001
  lr_critic: 0.001
  gamma: 0.99
  tau: 0.005
  hidden: 256
  buffer_capacity: 100000
  normalize_obs: true
  normalize_rew: true
  # TD3 / DDPG only
  policy_delay: 2
  noise:
    kind: decay        # gaussian | ornstein-uhlenbeck | decay
    ...
  # SAC only
  lr_alpha: 0.0003
  init_alpha: 0.2
  auto_tune_alpha: true

training:
  num_episodes: 2000
  warmup_steps: 1000
  batch_size: 64
  save_every: 500
  wandb: false

output_dir: outputs
device: cpu
```

## Project structure

```
configs/
  single_drone.yaml     single-drone training config
  swarm.yaml            swarm training config
  test.yaml             fast CI smoke-test config

scripts/
  train.py              training entry point
  evaluate.py           checkpoint evaluation

src/drone_rl/
  agents/
    base.py             AgentBase ABC + RunningMeanStd
    td3.py              TD3Agent
    ddpg.py             DDPGAgent
    sac.py              SACAgent
    factory.py          make_agent() — routes algo name to class
    noise.py            Gaussian / OU / decay noise factory
    replay_buffer.py    uniform experience replay
  envs/
    drone_env.py        DroneEnv (single drone, Gymnasium)
    swarm_env.py        SwarmEnv (N drones, centralized control)
  rewards/
    coverage.py         CoverageReward (grid-based)
  utils/
    logger.py           TrainingLogger (W&B + CSV)
    hardware.py         TelloEnv — real Tello drop-in for DroneEnv
    visualization.py    trajectory rendering helpers

tests/
  test_drone_env.py     DroneEnv unit tests
  test_radar_obs.py     radar_obs property tests
  test_agents.py        parametrized contract tests (all algos)
  test_td3.py           TD3-specific tests
  test_factory.py       make_agent routing tests
  test_logger.py        TrainingLogger CSV / W&B tests
  test_smoke.py         end-to-end training loop (all algos × modes)
```

## Experiment tracking

Training logs are always written to `outputs/train_log.csv`. Enable W&B with `--wandb` or `training.wandb: true` in config.

Logged metrics per episode: `episode_reward`, `episode_length`, `coverage_fraction`, `steps_per_second`, `critic_loss`, `actor_loss`, `mean_q`. SAC additionally logs `alpha` and `alpha_loss`.

## Sim-to-real

`TelloEnv` in `src/drone_rl/utils/hardware.py` implements the same `reset()` / `step()` interface as `DroneEnv` and returns the same 11-dim observation (physics dims from Tello SDK telemetry, radar dims zero-padded until a radar station is wired in). Swap `DroneEnv` for `TelloEnv` at deploy time — the agent is untouched.

Hardware extras required:

```bash
pip install -e ".[hardware]"
```

Connect the laptop to the Tello WiFi SSID (`TELLO-XXXXXX`) before calling `TelloEnv.reset()`.

## Development

```bash
pytest                   # run all tests (83)
pytest tests/test_smoke.py -v   # smoke tests only
black src/ tests/ scripts/
flake8 src/ tests/ scripts/ --max-line-length 100
```
