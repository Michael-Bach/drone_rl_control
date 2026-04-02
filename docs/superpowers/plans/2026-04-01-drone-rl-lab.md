# Drone RL Lab — Multi-Algorithm Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the single-algorithm TD3 system into a pluggable RL lab supporting TD3, SAC, and DDPG, with radar observation (OBS_DIM 7→11) and W&B + CSV experiment tracking.

**Architecture:** `AgentBase` ABC defines `select_action / store / train_step / save / load`; each algorithm lives in its own file and extends it; `make_agent(cfg, obs_dim, action_dim)` factory routes by `cfg["agent"]["type"]` (overridable via `--algo` CLI). `DroneEnv` gains a `radar_obs: float32[4]` attribute appended to the observation vector. `TrainingLogger` wraps W&B and CSV behind a single `log(episode, step, metrics)` call.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, Gymnasium 0.29+, NumPy, PyYAML, wandb (optional), pytest, tqdm

---

## File Map

**Created:**
- `src/drone_rl/agents/base.py` — `AgentBase` ABC + `RunningMeanStd` (moved from td3.py)
- `src/drone_rl/agents/ddpg.py` — `DDPGAgent(AgentBase)`
- `src/drone_rl/agents/sac.py` — `SACAgent(AgentBase)`
- `src/drone_rl/agents/factory.py` — `make_agent(cfg, obs_dim, action_dim) -> AgentBase`
- `src/drone_rl/utils/logger.py` — `TrainingLogger`
- `tests/test_radar_obs.py` — radar observation tests
- `tests/test_agents.py` — AgentBase contract tests for all three algorithms
- `tests/test_factory.py` — factory routing tests

**Modified:**
- `src/drone_rl/envs/drone_env.py` — `OBS_DIM = 11`, add `radar_obs` attribute
- `src/drone_rl/agents/td3.py` — extend `AgentBase`, rename methods, import `RunningMeanStd` from base
- `tests/test_td3.py` — update method names to match refactored TD3Agent
- `scripts/train.py` — add `--algo` / `--wandb` flags, use factory + logger
- `tests/test_smoke.py` — extend with per-algo smoke runs
- `configs/single_drone.yaml`, `configs/swarm.yaml`, `configs/test.yaml` — add `agent.type` and SAC keys
- `pyproject.toml` — add `wandb>=0.16` to optional deps

---

## Task 1: Extend DroneEnv with Radar Observation

**Files:**
- Modify: `src/drone_rl/envs/drone_env.py`
- Create: `tests/test_radar_obs.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_radar_obs.py
from pathlib import Path
import numpy as np
import yaml
from drone_rl.envs.drone_env import DroneEnv, OBS_DIM, ACTION_DIM


def _cfg():
    with open(Path(__file__).parent.parent / "configs" / "test.yaml") as f:
        return yaml.safe_load(f)


def test_obs_dim_is_11():
    assert OBS_DIM == 11


def test_obs_shape_is_11():
    env = DroneEnv(_cfg())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (11,)


def test_observation_space_shape_is_11():
    env = DroneEnv(_cfg())
    assert env.observation_space.shape == (11,)


def test_radar_obs_default_zeros():
    env = DroneEnv(_cfg())
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    np.testing.assert_array_equal(obs[7:], [0.0, 0.0, 0.0, 0.0])


def test_radar_obs_set_externally_appears_in_obs():
    env = DroneEnv(_cfg())
    env.reset(seed=0)
    env.radar_obs = np.array([1.0, 2.0, 3.0, 100.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    np.testing.assert_array_equal(obs[7:], [1.0, 2.0, 3.0, 100.0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /path/to/drone-rl-control
pytest tests/test_radar_obs.py -v
```

Expected: 5 failures (OBS_DIM is 7, radar_obs doesn't exist yet).

- [ ] **Step 3: Implement the changes in drone_env.py**

Change line 19 (`OBS_DIM = 7`) to:
```python
OBS_DIM = 11  # x, y, z, vx, vy, vz, yaw, radar_x, radar_y, radar_z, radar_t
```

In `DroneEnv.__init__`, after the existing mutable state block (after `self._step_count = 0`, around line 77), add:
```python
        self.radar_obs: np.ndarray = np.zeros(4, dtype=np.float32)
```

Update the docstring on line 27:
```python
    Observation: [x, y, z, vx, vy, vz, yaw, radar_x, radar_y, radar_z, radar_t]  (11-dim, float32)
```

Replace the `_obs` method (lines 152-157):
```python
    def _obs(self) -> np.ndarray:
        return np.concatenate([
            np.array(
                [self._x, self._y, self._z,
                 self._vx, self._vy, self._vz, self._yaw],
                dtype=np.float32,
            ),
            self.radar_obs,
        ])
```

Also update the `SwarmEnv` docstring in `src/drone_rl/envs/swarm_env.py`, line 19, to reflect the new obs dim:
```python
    Observation: (N * 11,) — concatenated drone states (7 physics + 4 radar each)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_radar_obs.py tests/test_drone_env.py tests/test_swarm_env.py -v
```

Expected: all pass. `test_drone_env.py` passes automatically because it imports `OBS_DIM` (not hardcoded 7).

- [ ] **Step 5: Commit**

```bash
git add src/drone_rl/envs/drone_env.py src/drone_rl/envs/swarm_env.py tests/test_radar_obs.py
git commit -m "feat: extend DroneEnv obs with radar_obs [x,y,z,t] (OBS_DIM 7→11)"
```

---

## Task 2: AgentBase ABC and Move RunningMeanStd

**Files:**
- Create: `src/drone_rl/agents/base.py`

- [ ] **Step 1: Write the file**

```python
# src/drone_rl/agents/base.py
"""Abstract base class for all off-policy RL agents, plus shared utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class RunningMeanStd:
    """Online mean/variance estimation (Welford's algorithm)."""

    def __init__(self, shape: tuple, eps: float = 1e-4) -> None:
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        b_mean, b_var, b_cnt = x.mean(0), x.var(0), x.shape[0]
        delta     = b_mean - self.mean
        tot_count = self.count + b_cnt
        new_mean  = self.mean + delta * b_cnt / tot_count
        M2 = (self.var * self.count + b_var * b_cnt
              + delta ** 2 * self.count * b_cnt / tot_count)
        self.mean, self.var, self.count = new_mean, M2 / tot_count, tot_count

    def normalize(self, x: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
        x = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        if clip is not None:
            x = np.clip(x, -clip, clip)
        return x.astype(np.float32)


class AgentBase(ABC):
    """Minimal interface all off-policy agents must satisfy."""

    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return a clipped action array for the given observation."""

    @abstractmethod
    def store(self, obs: np.ndarray, action: np.ndarray, reward: float,
              next_obs: np.ndarray, done: float) -> None:
        """Store a transition and update normalisation statistics."""

    @abstractmethod
    def train_step(self) -> dict:
        """One gradient update. Returns a metric dict (empty if buffer not ready)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist agent state to file at path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore agent state from file at path."""
```

- [ ] **Step 2: Verify the file is importable**

```bash
cd /path/to/drone-rl-control
python -c "from drone_rl.agents.base import AgentBase, RunningMeanStd; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/drone_rl/agents/base.py
git commit -m "feat: add AgentBase ABC and move RunningMeanStd to agents/base.py"
```

---

## Task 3: Refactor TD3Agent to Extend AgentBase

**Files:**
- Modify: `src/drone_rl/agents/td3.py`
- Modify: `tests/test_td3.py`

Key changes:
- `RunningMeanStd` imported from `base.py` instead of defined locally
- `class TD3Agent(AgentBase):`
- `batch_size` added to constructor
- `train(batch_size)` → `train_step() -> dict`
- `select_action(obs)` gains `deterministic=False` parameter (replaces `select_action_deterministic`)
- `select_action_deterministic` removed

- [ ] **Step 1: Update tests/test_td3.py to reflect the new API (write updated test file)**

Replace the entire `tests/test_td3.py`:
```python
from pathlib import Path
import numpy as np
import pytest
import torch
from drone_rl.agents.replay_buffer import ReplayBuffer
from drone_rl.agents.noise import make_noise
from drone_rl.agents.td3 import Actor, Critic, TD3Agent

STATE_DIM  = 7
ACTION_DIM = 4


# ---- ReplayBuffer ----

def test_replay_buffer_add_and_sample():
    buf = ReplayBuffer(capacity=100)
    for _ in range(20):
        buf.add(
            np.zeros(STATE_DIM), np.zeros(ACTION_DIM),
            0.0, np.zeros(STATE_DIM), 0.0,
        )
    s, a, r, ns, d = buf.sample(10)
    assert s.shape  == (10, STATE_DIM)
    assert a.shape  == (10, ACTION_DIM)
    assert r.shape  == (10, 1)
    assert ns.shape == (10, STATE_DIM)
    assert d.shape  == (10, 1)


def test_replay_buffer_is_ready():
    buf = ReplayBuffer(capacity=100)
    assert not buf.is_ready(10)
    for _ in range(10):
        buf.add(np.zeros(STATE_DIM), np.zeros(ACTION_DIM),
                0.0, np.zeros(STATE_DIM), 0.0)
    assert buf.is_ready(10)


# ---- Noise ----

def test_gaussian_noise_shape():
    noise = make_noise({"kind": "gaussian", "dim": ACTION_DIM, "sigma": 0.1})
    out = noise()
    assert out.shape == (ACTION_DIM,)


def test_decay_noise_decreases():
    noise = make_noise({
        "kind": "decay", "dim": ACTION_DIM, "per": "step",
        "gamma": 0.5, "scale_min": 0.01,
        "base": {"kind": "gaussian", "dim": ACTION_DIM, "sigma": 1.0},
    })
    first = np.abs(noise()).mean()
    for _ in range(100):
        noise()
    later = np.abs(noise()).mean()
    assert later <= first + 0.1


# ---- Actor / Critic ----

def test_actor_output_shape():
    actor = Actor(STATE_DIM, ACTION_DIM, hidden=64)
    obs = torch.zeros(1, STATE_DIM)
    out = actor(obs)
    assert out.shape == (1, ACTION_DIM)


def test_actor_output_in_tanh_range():
    actor = Actor(STATE_DIM, ACTION_DIM, hidden=64)
    obs = torch.randn(32, STATE_DIM)
    out = actor(obs)
    assert out.abs().max().item() <= 1.0 + 1e-6


def test_critic_output_shape():
    critic = Critic(STATE_DIM, ACTION_DIM, hidden=64)
    obs    = torch.zeros(1, STATE_DIM)
    action = torch.zeros(1, ACTION_DIM)
    out = critic(obs, action)
    assert out.shape == (1, 1)


# ---- TD3Agent ----

@pytest.fixture
def agent():
    return TD3Agent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        noise_cfg={"kind": "gaussian", "dim": ACTION_DIM, "sigma": 0.1},
        hidden=64, buffer_capacity=200, batch_size=10, device="cpu",
    )


def test_select_action_shape(agent):
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    action = agent.select_action(obs)
    assert action.shape == (ACTION_DIM,)


def test_select_action_clipped(agent):
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    action = agent.select_action(obs)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_select_action_deterministic_no_noise(agent):
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    # Two deterministic calls must return identical results
    a1 = agent.select_action(obs, deterministic=True)
    a2 = agent.select_action(obs, deterministic=True)
    np.testing.assert_array_equal(a1, a2)


def test_train_step_empty_when_buffer_not_ready(agent):
    result = agent.train_step()
    assert result == {}


def test_train_step_returns_metric_dict(agent):
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    for _ in range(50):
        agent.store(obs, np.zeros(ACTION_DIM), 1.0, obs, False)
    result = agent.train_step()
    assert isinstance(result, dict)
    assert "critic_loss" in result
    assert "actor_loss" in result


def test_save_and_load(agent, tmp_path):
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    for _ in range(20):
        agent.store(obs, np.zeros(ACTION_DIM), 1.0, obs, False)
    agent.train_step()
    path = str(tmp_path / "checkpoint.pt")
    agent.save(path)
    agent.load(path)
    action = agent.select_action(obs, deterministic=True)
    assert action.shape == (ACTION_DIM,)
```

- [ ] **Step 2: Run updated tests to verify they fail on the old API**

```bash
pytest tests/test_td3.py -v
```

Expected: failures on `train_step`, `select_action(obs, deterministic=True)`, `batch_size` kwarg.

- [ ] **Step 3: Rewrite src/drone_rl/agents/td3.py**

```python
"""Twin Delayed DDPG (TD3) agent."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drone_rl.agents.base import AgentBase, RunningMeanStd
from drone_rl.agents.noise import BaseNoise, make_noise
from drone_rl.agents.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


# ---------------------------------------------------------------------------
# TD3 Agent
# ---------------------------------------------------------------------------

class TD3Agent(AgentBase):
    """
    Twin Delayed DDPG with optional obs/reward normalisation.

    Parameters
    ----------
    state_dim       : observation vector length
    action_dim      : action vector length
    noise_cfg       : config dict passed to make_noise()
    lr_actor        : actor learning rate
    lr_critic       : critic learning rate
    gamma           : discount factor
    tau             : Polyak averaging coefficient
    policy_delay    : actor update frequency (relative to critic updates)
    policy_noise    : std of target policy smoothing noise
    noise_clip      : clamp range for smoothing noise
    hidden          : hidden layer width
    buffer_capacity : replay buffer capacity
    batch_size      : training batch size
    normalize_obs   : enable online obs normalisation
    normalize_rew   : enable online reward normalisation
    grad_clip       : max gradient norm
    device          : torch device string
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        noise_cfg: Dict[str, Any],
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        policy_noise: float = 0.1,
        noise_clip: float = 0.3,
        hidden: int = 256,
        buffer_capacity: int = 200_000,
        batch_size: int = 64,
        normalize_obs: bool = True,
        normalize_rew: bool = True,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.device       = torch.device(device)
        self.gamma        = gamma
        self.tau          = tau
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.grad_clip    = grad_clip
        self.batch_size   = batch_size
        self.normalize_obs = normalize_obs
        self.normalize_rew = normalize_rew
        self._train_step  = 0

        self.actor          = Actor(state_dim, action_dim, hidden).to(self.device)
        self.actor_target   = deepcopy(self.actor)
        self.critic1        = Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2        = Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic2_target = deepcopy(self.critic2)

        self.actor_opt   = torch.optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.noise: BaseNoise = make_noise(noise_cfg)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        self.rew_rms = RunningMeanStd(shape=())

    # ------------------------------------------------------------------
    # AgentBase interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.normalize_obs:
            obs = self.obs_rms.normalize(obs)
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(t).cpu().numpy()[0]
        if deterministic:
            return np.clip(a, -1.0, 1.0).astype(np.float32)
        return np.clip(a + self.noise(), -1.0, 1.0).astype(np.float32)

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float,
              next_obs: np.ndarray, done: float) -> None:
        if self.normalize_obs:
            self.obs_rms.update(np.array([obs]))
        if self.normalize_rew:
            self.rew_rms.update(np.array([reward]))
        self.buffer.add(obs, action, reward, next_obs, done)

    def train_step(self) -> dict:
        if not self.buffer.is_ready(self.batch_size):
            return {}

        states, actions, rewards, next_s, dones = self.buffer.sample(self.batch_size)
        states  = states.to(self.device)
        actions = actions.to(self.device)
        next_s  = next_s.to(self.device)
        dones   = dones.to(self.device)

        if self.normalize_rew:
            r_np = self.rew_rms.normalize(rewards.squeeze(1).numpy(), clip=5.0)
            rewards = torch.as_tensor(r_np, dtype=torch.float32).unsqueeze(1)
        rewards = rewards.to(self.device)

        with torch.no_grad():
            noise  = (torch.randn_like(actions) * self.policy_noise
                      ).clamp(-self.noise_clip, self.noise_clip)
            next_a = (self.actor_target(next_s) + noise).clamp(-1.0, 1.0)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(
                self.critic1_target(next_s, next_a),
                self.critic2_target(next_s, next_a),
            )

        c_losses = []
        for critic, opt in ((self.critic1, self.critic1_opt),
                            (self.critic2, self.critic2_opt)):
            loss = F.smooth_l1_loss(critic(states, actions), target_q)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
            opt.step()
            c_losses.append(loss.item())

        critic_loss = sum(c_losses) / len(c_losses)
        mean_q      = self.critic1(states, actions).mean().item()
        actor_loss  = 0.0

        if self._train_step % self.policy_delay == 0:
            pa     = self.actor(states)
            a_loss = -self.critic1(states, pa).mean() + 1e-3 * (pa ** 2).mean()
            self.actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_opt.step()
            for tgt, src in ((self.actor_target, self.actor),
                             (self.critic1_target, self.critic1),
                             (self.critic2_target, self.critic2)):
                self._soft_update(tgt, src, self.tau)
            actor_loss = a_loss.item()

        self._train_step += 1
        return {"critic_loss": critic_loss, "actor_loss": actor_loss, "mean_q": mean_q}

    def save(self, path: str) -> None:
        torch.save({
            "actor":           self.actor.state_dict(),
            "actor_target":    self.actor_target.state_dict(),
            "critic1":         self.critic1.state_dict(),
            "critic1_target":  self.critic1_target.state_dict(),
            "critic2":         self.critic2.state_dict(),
            "critic2_target":  self.critic2_target.state_dict(),
            "obs_rms_mean":    self.obs_rms.mean,
            "obs_rms_var":     self.obs_rms.var,
            "rew_rms_mean":    self.rew_rms.mean,
            "rew_rms_var":     self.rew_rms.var,
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.actor_target.load_state_dict(ck["actor_target"])
        self.critic1.load_state_dict(ck["critic1"])
        self.critic2.load_state_dict(ck["critic2"])
        if "critic1_target" in ck:
            self.critic1_target.load_state_dict(ck["critic1_target"])
            self.critic2_target.load_state_dict(ck["critic2_target"])
        if "obs_rms_mean" in ck:
            self.obs_rms.mean = ck["obs_rms_mean"]
            self.obs_rms.var  = ck["obs_rms_var"]
        if "rew_rms_mean" in ck:
            self.rew_rms.mean = ck["rew_rms_mean"]
            self.rew_rms.var  = ck["rew_rms_var"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_td3.py -v
```

Expected: all pass.

- [ ] **Step 5: Run full suite to catch regressions**

```bash
pytest -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/drone_rl/agents/td3.py tests/test_td3.py
git commit -m "refactor: TD3Agent extends AgentBase, train_step()->dict, unified select_action"
```

---

## Task 4: Implement DDPGAgent

**Files:**
- Create: `src/drone_rl/agents/ddpg.py`

DDPG has a single critic (no twin), single target pair, and OUNoise. `train_step()` updates both networks every call (no policy delay).

- [ ] **Step 1: Write src/drone_rl/agents/ddpg.py**

```python
"""DDPG agent — single critic, OUNoise exploration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drone_rl.agents.base import AgentBase, RunningMeanStd
from drone_rl.agents.noise import BaseNoise, make_noise
from drone_rl.agents.replay_buffer import ReplayBuffer


class _Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


class DDPGAgent(AgentBase):
    """DDPG: single critic, Polyak targets, OUNoise exploration."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        noise_cfg: Dict[str, Any],
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden: int = 256,
        buffer_capacity: int = 200_000,
        batch_size: int = 64,
        normalize_obs: bool = True,
        normalize_rew: bool = True,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.device        = torch.device(device)
        self.gamma         = gamma
        self.tau           = tau
        self.grad_clip     = grad_clip
        self.batch_size    = batch_size
        self.normalize_obs = normalize_obs
        self.normalize_rew = normalize_rew

        self.actor         = _Actor(state_dim, action_dim, hidden).to(self.device)
        self.actor_target  = deepcopy(self.actor)
        self.critic        = _Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic_target = deepcopy(self.critic)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.noise: BaseNoise = make_noise(noise_cfg)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        self.rew_rms = RunningMeanStd(shape=())

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.normalize_obs:
            obs = self.obs_rms.normalize(obs)
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(t).cpu().numpy()[0]
        if deterministic:
            return np.clip(a, -1.0, 1.0).astype(np.float32)
        return np.clip(a + self.noise(), -1.0, 1.0).astype(np.float32)

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float,
              next_obs: np.ndarray, done: float) -> None:
        if self.normalize_obs:
            self.obs_rms.update(np.array([obs]))
        if self.normalize_rew:
            self.rew_rms.update(np.array([reward]))
        self.buffer.add(obs, action, reward, next_obs, done)

    def train_step(self) -> dict:
        if not self.buffer.is_ready(self.batch_size):
            return {}

        states, actions, rewards, next_s, dones = self.buffer.sample(self.batch_size)
        states  = states.to(self.device)
        actions = actions.to(self.device)
        next_s  = next_s.to(self.device)
        dones   = dones.to(self.device)

        if self.normalize_rew:
            r_np    = self.rew_rms.normalize(rewards.squeeze(1).numpy(), clip=5.0)
            rewards = torch.as_tensor(r_np, dtype=torch.float32).unsqueeze(1)
        rewards = rewards.to(self.device)

        with torch.no_grad():
            next_a   = self.actor_target(next_s).clamp(-1.0, 1.0)
            target_q = rewards + self.gamma * (1 - dones) * self.critic_target(next_s, next_a)

        critic_loss = F.smooth_l1_loss(self.critic(states, actions), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        for tgt, src in ((self.actor_target, self.actor), (self.critic_target, self.critic)):
            for tp, sp in zip(tgt.parameters(), src.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        mean_q = self.critic(states, actions).detach().mean().item()
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "mean_q":      mean_q,
        }

    def save(self, path: str) -> None:
        torch.save({
            "actor":          self.actor.state_dict(),
            "actor_target":   self.actor_target.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "obs_rms_mean":   self.obs_rms.mean,
            "obs_rms_var":    self.obs_rms.var,
            "rew_rms_mean":   self.rew_rms.mean,
            "rew_rms_var":    self.rew_rms.var,
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.actor_target.load_state_dict(ck["actor_target"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_target.load_state_dict(ck["critic_target"])
        if "obs_rms_mean" in ck:
            self.obs_rms.mean = ck["obs_rms_mean"]
            self.obs_rms.var  = ck["obs_rms_var"]
        if "rew_rms_mean" in ck:
            self.rew_rms.mean = ck["rew_rms_mean"]
            self.rew_rms.var  = ck["rew_rms_var"]
```

- [ ] **Step 2: Verify it is importable**

```bash
python -c "from drone_rl.agents.ddpg import DDPGAgent; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/drone_rl/agents/ddpg.py
git commit -m "feat: add DDPGAgent (single critic, OUNoise, extends AgentBase)"
```

---

## Task 5: Implement SACAgent

**Files:**
- Create: `src/drone_rl/agents/sac.py`

SAC uses a squashed-Gaussian actor, twin critics, and auto-tunes the entropy temperature α. No external exploration noise.

- [ ] **Step 1: Write src/drone_rl/agents/sac.py**

```python
"""SAC agent — squashed Gaussian actor, auto-entropy tuning."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drone_rl.agents.base import AgentBase, RunningMeanStd
from drone_rl.agents.replay_buffer import ReplayBuffer


class _SACGaussianActor(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor):
        h       = self.net(x)
        mean    = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor):
        mean, log_std = self(x)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u    = dist.rsample()
        action   = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def mean_action(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self(x)
        return torch.tanh(mean)


class _Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


class SACAgent(AgentBase):
    """SAC with twin critics and automatic entropy tuning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        init_alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden: int = 256,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        normalize_obs: bool = True,
        normalize_rew: bool = True,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.device          = torch.device(device)
        self.gamma           = gamma
        self.tau             = tau
        self.grad_clip       = grad_clip
        self.batch_size      = batch_size
        self.normalize_obs   = normalize_obs
        self.normalize_rew   = normalize_rew
        self.auto_tune_alpha = auto_tune_alpha
        self.target_entropy  = float(-action_dim)

        self.actor          = _SACGaussianActor(state_dim, action_dim, hidden).to(self.device)
        self.critic1        = _Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2        = _Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic2_target = deepcopy(self.critic2)

        self.actor_opt   = torch.optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32,
            requires_grad=True, device=self.device,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.buffer  = ReplayBuffer(buffer_capacity)
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        self.rew_rms = RunningMeanStd(shape=())

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.normalize_obs:
            obs = self.obs_rms.normalize(obs)
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.actor.mean_action(t).cpu().numpy()[0]
            else:
                a, _ = self.actor.sample(t)
                a = a.cpu().numpy()[0]
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float,
              next_obs: np.ndarray, done: float) -> None:
        if self.normalize_obs:
            self.obs_rms.update(np.array([obs]))
        if self.normalize_rew:
            self.rew_rms.update(np.array([reward]))
        self.buffer.add(obs, action, reward, next_obs, done)

    def train_step(self) -> dict:
        if not self.buffer.is_ready(self.batch_size):
            return {}

        states, actions, rewards, next_s, dones = self.buffer.sample(self.batch_size)
        states  = states.to(self.device)
        actions = actions.to(self.device)
        next_s  = next_s.to(self.device)
        dones   = dones.to(self.device)

        if self.normalize_rew:
            r_np    = self.rew_rms.normalize(rewards.squeeze(1).numpy(), clip=5.0)
            rewards = torch.as_tensor(r_np, dtype=torch.float32).unsqueeze(1)
        rewards = rewards.to(self.device)

        # ---- Critic update ----
        with torch.no_grad():
            next_a, next_log_pi = self.actor.sample(next_s)
            target_q = rewards + self.gamma * (1 - dones) * (
                torch.min(self.critic1_target(next_s, next_a),
                          self.critic2_target(next_s, next_a))
                - self.log_alpha.exp() * next_log_pi
            )

        c_losses = []
        for critic, opt in ((self.critic1, self.critic1_opt),
                             (self.critic2, self.critic2_opt)):
            loss = F.smooth_l1_loss(critic(states, actions), target_q)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
            opt.step()
            c_losses.append(loss.item())

        # ---- Actor update ----
        pi, log_pi = self.actor.sample(states)
        actor_loss = (self.log_alpha.exp() * log_pi
                      - torch.min(self.critic1(states, pi),
                                  self.critic2(states, pi))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        # ---- Alpha update ----
        alpha_loss_val = 0.0
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = alpha_loss.item()

        # ---- Soft target update ----
        for tgt, src in ((self.critic1_target, self.critic1),
                          (self.critic2_target, self.critic2)):
            for tp, sp in zip(tgt.parameters(), src.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return {
            "critic_loss": sum(c_losses) / len(c_losses),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
            "alpha_loss":  alpha_loss_val,
            "mean_q":      self.critic1(states, actions).detach().mean().item(),
        }

    def save(self, path: str) -> None:
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic1":        self.critic1.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2":        self.critic2.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha":      self.log_alpha.detach().cpu(),
            "obs_rms_mean":   self.obs_rms.mean,
            "obs_rms_var":    self.obs_rms.var,
            "rew_rms_mean":   self.rew_rms.mean,
            "rew_rms_var":    self.rew_rms.var,
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic1.load_state_dict(ck["critic1"])
        self.critic1_target.load_state_dict(ck["critic1_target"])
        self.critic2.load_state_dict(ck["critic2"])
        self.critic2_target.load_state_dict(ck["critic2_target"])
        if "log_alpha" in ck:
            with torch.no_grad():
                self.log_alpha.copy_(ck["log_alpha"].to(self.device))
        if "obs_rms_mean" in ck:
            self.obs_rms.mean = ck["obs_rms_mean"]
            self.obs_rms.var  = ck["obs_rms_var"]
        if "rew_rms_mean" in ck:
            self.rew_rms.mean = ck["rew_rms_mean"]
            self.rew_rms.var  = ck["rew_rms_var"]
```

- [ ] **Step 2: Verify importable**

```bash
python -c "from drone_rl.agents.sac import SACAgent; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/drone_rl/agents/sac.py
git commit -m "feat: add SACAgent (squashed Gaussian, auto-entropy, extends AgentBase)"
```

---

## Task 6: Agent Factory + Contract Tests

**Files:**
- Create: `src/drone_rl/agents/factory.py`
- Create: `tests/test_agents.py`
- Create: `tests/test_factory.py`

- [ ] **Step 1: Write tests/test_agents.py**

```python
"""AgentBase contract tests: all three agents must satisfy the interface."""
from pathlib import Path
import numpy as np
import pytest
import yaml
from drone_rl.agents.td3 import TD3Agent
from drone_rl.agents.ddpg import DDPGAgent
from drone_rl.agents.sac import SACAgent

STATE_DIM  = 11  # matches new OBS_DIM
ACTION_DIM = 4

_NOISE_CFG = {"kind": "gaussian", "dim": ACTION_DIM, "sigma": 0.1}


def _make_td3():
    return TD3Agent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        noise_cfg=_NOISE_CFG, hidden=64,
        buffer_capacity=200, batch_size=10, device="cpu",
    )


def _make_ddpg():
    return DDPGAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        noise_cfg=_NOISE_CFG, hidden=64,
        buffer_capacity=200, batch_size=10, device="cpu",
    )


def _make_sac():
    return SACAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        hidden=64, buffer_capacity=200, batch_size=10, device="cpu",
    )


@pytest.mark.parametrize("make_agent", [_make_td3, _make_ddpg, _make_sac])
def test_select_action_shape(make_agent):
    agent = make_agent()
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    action = agent.select_action(obs)
    assert action.shape == (ACTION_DIM,)


@pytest.mark.parametrize("make_agent", [_make_td3, _make_ddpg, _make_sac])
def test_select_action_clipped(make_agent):
    agent = make_agent()
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    action = agent.select_action(obs)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


@pytest.mark.parametrize("make_agent", [_make_td3, _make_ddpg, _make_sac])
def test_train_step_empty_when_not_ready(make_agent):
    agent = make_agent()
    assert agent.train_step() == {}


@pytest.mark.parametrize("make_agent", [_make_td3, _make_ddpg, _make_sac])
def test_train_step_returns_dict_with_keys(make_agent):
    agent = make_agent()
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    for _ in range(50):
        agent.store(obs, np.zeros(ACTION_DIM), 1.0, obs, 0.0)
    result = agent.train_step()
    assert isinstance(result, dict)
    assert "critic_loss" in result
    assert "actor_loss" in result


@pytest.mark.parametrize("make_agent", [_make_td3, _make_ddpg, _make_sac])
def test_save_load_roundtrip(make_agent, tmp_path):
    agent = make_agent()
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    for _ in range(20):
        agent.store(obs, np.zeros(ACTION_DIM), 1.0, obs, 0.0)
    agent.train_step()
    path = str(tmp_path / "ckpt.pt")
    agent.save(path)
    agent.load(path)
    action = agent.select_action(obs, deterministic=True)
    assert action.shape == (ACTION_DIM,)
```

- [ ] **Step 2: Write tests/test_factory.py**

```python
"""Factory routing and error handling."""
import pytest
from drone_rl.agents.factory import make_agent
from drone_rl.agents.td3 import TD3Agent
from drone_rl.agents.ddpg import DDPGAgent
from drone_rl.agents.sac import SACAgent

_BASE_CFG = {
    "agent": {
        "type": "td3",
        "lr_actor": 1e-4, "lr_critic": 1e-3,
        "gamma": 0.99, "tau": 0.005,
        "hidden": 64, "buffer_capacity": 200,
        "normalize_obs": True, "normalize_rew": True,
        "policy_delay": 2,
        "noise": {"kind": "gaussian", "dim": 4, "sigma": 0.1},
        "lr_alpha": 3e-4, "init_alpha": 0.2, "auto_tune_alpha": True,
    },
    "training": {"batch_size": 10},
    "device": "cpu",
}


def _cfg(algo: str) -> dict:
    import copy
    c = copy.deepcopy(_BASE_CFG)
    c["agent"]["type"] = algo
    return c


def test_make_agent_td3():
    agent = make_agent(_cfg("td3"), obs_dim=11, action_dim=4)
    assert isinstance(agent, TD3Agent)


def test_make_agent_ddpg():
    agent = make_agent(_cfg("ddpg"), obs_dim=11, action_dim=4)
    assert isinstance(agent, DDPGAgent)


def test_make_agent_sac():
    agent = make_agent(_cfg("sac"), obs_dim=11, action_dim=4)
    assert isinstance(agent, SACAgent)


def test_make_agent_unknown_raises():
    with pytest.raises(ValueError, match="Unknown agent type"):
        make_agent(_cfg("ppo"), obs_dim=11, action_dim=4)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_agents.py tests/test_factory.py -v
```

Expected: `ModuleNotFoundError: drone_rl.agents.factory` (file doesn't exist yet).

- [ ] **Step 4: Write src/drone_rl/agents/factory.py**

```python
"""Agent factory: instantiate the right agent from config."""

from __future__ import annotations

from typing import Any, Dict

from drone_rl.agents.base import AgentBase


def make_agent(cfg: Dict[str, Any], obs_dim: int, action_dim: int) -> AgentBase:
    """
    Instantiate an agent from config.

    Reads cfg["agent"]["type"] (td3 | sac | ddpg).
    cfg["training"]["batch_size"] is used as the default batch size.
    """
    ag   = cfg["agent"]
    algo = str(ag.get("type", "td3")).lower()

    shared = dict(
        state_dim=obs_dim,
        action_dim=action_dim,
        lr_actor=float(ag.get("lr_actor", 3e-4)),
        lr_critic=float(ag.get("lr_critic", 3e-4)),
        gamma=float(ag.get("gamma", 0.99)),
        tau=float(ag.get("tau", 0.005)),
        hidden=int(ag.get("hidden", 256)),
        buffer_capacity=int(ag.get("buffer_capacity", 1_000_000)),
        batch_size=int(ag.get("batch_size", cfg.get("training", {}).get("batch_size", 256))),
        normalize_obs=bool(ag.get("normalize_obs", True)),
        normalize_rew=bool(ag.get("normalize_rew", True)),
        device=str(cfg.get("device", "cpu")),
    )

    def _noise_cfg() -> dict:
        noise = dict(ag["noise"])
        noise["dim"] = action_dim
        if "base" in noise:
            noise["base"] = dict(noise["base"])
            noise["base"]["dim"] = action_dim
        return noise

    if algo == "td3":
        from drone_rl.agents.td3 import TD3Agent
        return TD3Agent(
            noise_cfg=_noise_cfg(),
            policy_delay=int(ag.get("policy_delay", 2)),
            **shared,
        )

    if algo == "ddpg":
        from drone_rl.agents.ddpg import DDPGAgent
        return DDPGAgent(noise_cfg=_noise_cfg(), **shared)

    if algo == "sac":
        from drone_rl.agents.sac import SACAgent
        return SACAgent(
            lr_alpha=float(ag.get("lr_alpha", 3e-4)),
            init_alpha=float(ag.get("init_alpha", 0.2)),
            auto_tune_alpha=bool(ag.get("auto_tune_alpha", True)),
            **shared,
        )

    raise ValueError(
        f"Unknown agent type {algo!r}. Valid options: td3, sac, ddpg"
    )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_agents.py tests/test_factory.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/drone_rl/agents/factory.py tests/test_agents.py tests/test_factory.py
git commit -m "feat: agent factory + AgentBase contract tests for TD3/DDPG/SAC"
```

---

## Task 7: TrainingLogger

**Files:**
- Create: `src/drone_rl/utils/logger.py`
- Create: `tests/test_logger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logger.py
import csv
from pathlib import Path
from drone_rl.utils.logger import TrainingLogger


def test_csv_written_with_episode_and_step(tmp_path):
    logger = TrainingLogger(
        run_name="test_run",
        output_dir=str(tmp_path),
        use_wandb=False,
    )
    logger.log(episode=1, step=10, metrics={"episode_reward": 5.0, "coverage_fraction": 0.3})
    logger.log(episode=2, step=25, metrics={"episode_reward": 8.0, "coverage_fraction": 0.5})
    logger.close()

    csv_path = tmp_path / "train_log.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 2
    assert rows[0]["episode"] == "1"
    assert rows[0]["episode_reward"] == "5.0"
    assert rows[1]["coverage_fraction"] == "0.5"


def test_logger_skips_wandb_when_disabled(tmp_path):
    # Should not raise even if wandb is not configured
    logger = TrainingLogger(run_name="nowandb", output_dir=str(tmp_path), use_wandb=False)
    logger.log(episode=1, step=1, metrics={"loss": 0.1})
    logger.close()


def test_logger_skips_wandb_gracefully_when_enabled_but_unconfigured(tmp_path):
    # use_wandb=True but no API key — must not raise
    logger = TrainingLogger(run_name="nowandb", output_dir=str(tmp_path), use_wandb=True)
    logger.log(episode=1, step=1, metrics={"loss": 0.1})
    logger.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_logger.py -v
```

Expected: `ModuleNotFoundError: drone_rl.utils.logger`

- [ ] **Step 3: Write src/drone_rl/utils/logger.py**

```python
"""Training logger: W&B + CSV behind a single interface."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    """
    Logs training metrics to CSV (always) and W&B (optional).

    Parameters
    ----------
    run_name      : experiment name used for W&B run and log filename
    output_dir    : directory where train_log.csv is written
    use_wandb     : whether to attempt W&B logging
    wandb_project : W&B project name (default: "drone-rl")
    cfg           : full training config dict, logged as W&B run config
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        cfg: Optional[Dict] = None,
    ) -> None:
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project or "drone-rl",
                    name=run_name,
                    config=cfg or {},
                )
                self._wandb = wandb
            except Exception:
                pass  # no API key or wandb not installed — silent skip

        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path   = log_dir / "train_log.csv"
        self._csv_file   = open(self._csv_path, "w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list] = None

    def log(self, episode: int, step: int, metrics: Dict[str, Any]) -> None:
        """Write one row to CSV (and W&B if enabled)."""
        row = {"episode": episode, "step": step, **metrics}

        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def close(self) -> None:
        """Flush and close all backends."""
        self._csv_file.close()
        if self._wandb is not None:
            self._wandb.finish()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_logger.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/drone_rl/utils/logger.py tests/test_logger.py
git commit -m "feat: TrainingLogger (W&B + CSV) with graceful W&B skip"
```

---

## Task 8: Refactor train.py

**Files:**
- Modify: `scripts/train.py`

Changes: remove TD3-specific import, use `make_agent`, add `--algo` and `--wandb` CLI flags, replace `agent.train(batch_size)` with `agent.train_step()`, add `TrainingLogger`, handle missing `noise` attribute on SAC.

- [ ] **Step 1: Replace scripts/train.py**

```python
#!/usr/bin/env python3
"""
train.py — Training entry point for drone RL.

Usage
-----
  python scripts/train.py --config configs/single_drone.yaml
  python scripts/train.py --config configs/single_drone.yaml --algo sac
  python scripts/train.py --config configs/swarm.yaml --swarm --algo td3
  python scripts/train.py --config configs/test.yaml --max-steps 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_rl.agents.factory import make_agent
from drone_rl.envs.drone_env import DroneEnv
from drone_rl.envs.swarm_env import SwarmEnv
from drone_rl.utils.logger import TrainingLogger


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--swarm",     action="store_true")
    parser.add_argument("--algo",      default=None,
                        help="Override agent.type in config (td3 | sac | ddpg)")
    parser.add_argument("--wandb",     action="store_true",
                        help="Enable W&B logging (overrides config)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override total env steps (for smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.algo:
        cfg["agent"]["type"] = args.algo
    if args.wandb:
        cfg.setdefault("training", {})["wandb"] = True

    tr  = cfg["training"]
    env = SwarmEnv(cfg) if args.swarm else DroneEnv(cfg)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent      = make_agent(cfg, obs_dim, action_dim)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_name = cfg["agent"].get("type", "td3")
    run_name  = f"{algo_name}_{'swarm' if args.swarm else 'single'}"
    logger    = TrainingLogger(
        run_name=run_name,
        output_dir=str(output_dir),
        use_wandb=bool(tr.get("wandb", False)),
        cfg=cfg,
    )

    warmup       = int(tr.get("warmup_steps", 1000))
    save_every   = int(tr.get("save_every", 500))
    num_episodes = int(tr.get("num_episodes", 1000))

    total_steps = 0
    max_steps   = args.max_steps  # None = run full training

    for ep in tqdm(range(num_episodes), desc="Episodes"):
        obs, _ = env.reset()
        if hasattr(agent, "noise"):
            agent.noise.reset()
        done = trunc = False
        ep_reward    = 0.0
        ep_start     = time.time()
        ep_steps     = 0
        info: dict   = {}
        train_metrics: dict = {}

        while not (done or trunc):
            if total_steps < warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, trunc, info = env.step(action)
            agent.store(obs, action, reward, next_obs, float(done))

            if total_steps >= warmup:
                train_metrics = agent.train_step()

            obs = next_obs
            ep_reward   += reward
            total_steps += 1
            ep_steps    += 1

            if max_steps is not None and total_steps >= max_steps:
                logger.close()
                return  # smoke test exit

        ep_time = time.time() - ep_start
        metrics = {
            "episode_reward":    ep_reward,
            "episode_length":    ep_steps,
            "coverage_fraction": info.get("coverage", 0.0),
            "steps_per_second":  ep_steps / max(ep_time, 1e-6),
        }
        metrics.update(train_metrics)
        logger.log(episode=ep + 1, step=total_steps, metrics=metrics)

        if (ep + 1) % save_every == 0:
            ckpt = output_dir / f"checkpoint_{ep + 1}.pt"
            agent.save(str(ckpt))
            tqdm.write(
                f"ep={ep+1}  reward={ep_reward:.2f}  "
                f"coverage={info.get('coverage', 0):.2%}  "
                f"saved={ckpt.name}"
            )

    final_path = output_dir / "checkpoint_final.pt"
    agent.save(str(final_path))
    logger.close()
    tqdm.write(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run existing smoke tests**

```bash
pytest tests/test_smoke.py -v
```

Expected: both single-drone and swarm smoke tests pass.

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "refactor: train.py uses agent factory + TrainingLogger, adds --algo/--wandb flags"
```

---

## Task 9: Update YAML Configs

**Files:**
- Modify: `configs/single_drone.yaml`
- Modify: `configs/swarm.yaml`
- Modify: `configs/test.yaml`

- [ ] **Step 1: Update configs/single_drone.yaml**

Replace the `agent:` section and add `wandb` to training:

```yaml
# configs/single_drone.yaml
env:
  x_max: 25.0
  y_max: 25.0
  max_altitude: 10.0
  max_speed: 8.0
  dt: 0.05
  max_steps: 500

reward:
  cell_size: 2.5
  coverage_bonus: 1.0
  revisit_penalty: 0.0
  boundary_penalty: 10.0

agent:
  type: td3  # td3 | sac | ddpg  (overridable via --algo)
  lr_actor: 0.0001
  lr_critic: 0.001
  gamma: 0.99
  tau: 0.005
  hidden: 256
  buffer_capacity: 100000
  normalize_obs: true
  normalize_rew: true
  # td3 / ddpg only
  policy_delay: 2
  noise:
    kind: decay
    dim: 4  # overridden at runtime by factory to match actual action_dim
    per: episode
    gamma: 0.995
    scale_min: 0.02
    base:
      kind: gaussian
      dim: 4  # overridden at runtime by factory to match actual action_dim
      sigma: 0.3
  # sac only
  lr_alpha: 0.0003
  init_alpha: 0.2
  auto_tune_alpha: true

training:
  num_episodes: 2000
  warmup_steps: 1000
  batch_size: 64
  save_every: 500
  rollout_every: 100
  wandb: false

output_dir: outputs
device: cpu
```

- [ ] **Step 2: Update configs/swarm.yaml**

```yaml
# configs/swarm.yaml
env:
  x_max: 50.0
  y_max: 50.0
  max_altitude: 10.0
  max_speed: 8.0
  dt: 0.05
  max_steps: 1000
  n_drones: 3
  min_separation: 2.0

reward:
  cell_size: 2.5
  coverage_bonus: 1.0
  revisit_penalty: 0.1
  boundary_penalty: 10.0
  proximity_penalty: 5.0

agent:
  type: td3  # td3 | sac | ddpg  (overridable via --algo)
  lr_actor: 0.0001
  lr_critic: 0.001
  gamma: 0.99
  tau: 0.005
  hidden: 256
  buffer_capacity: 200000
  normalize_obs: true
  normalize_rew: true
  # td3 / ddpg only
  policy_delay: 2
  noise:
    kind: decay
    dim: 12  # overridden at runtime by factory to match actual action_dim
    per: episode
    gamma: 0.995
    scale_min: 0.02
    base:
      kind: gaussian
      dim: 12  # overridden at runtime by factory to match actual action_dim
      sigma: 0.3
  # sac only
  lr_alpha: 0.0003
  init_alpha: 0.2
  auto_tune_alpha: true

training:
  num_episodes: 3000
  warmup_steps: 2000
  batch_size: 128
  save_every: 500
  rollout_every: 200
  wandb: false

output_dir: outputs
device: cpu
```

- [ ] **Step 3: Update configs/test.yaml**

```yaml
# configs/test.yaml — tiny env for fast CI
env:
  x_max: 10.0
  y_max: 10.0
  max_altitude: 5.0
  max_speed: 8.0
  dt: 0.05
  max_steps: 20
  n_drones: 2
  min_separation: 1.0

reward:
  cell_size: 2.0
  coverage_bonus: 1.0
  revisit_penalty: 0.0
  boundary_penalty: 10.0
  proximity_penalty: 5.0

agent:
  type: td3  # overridden per-test via --algo
  lr_actor: 0.001
  lr_critic: 0.001
  gamma: 0.99
  tau: 0.005
  hidden: 64
  buffer_capacity: 1000
  normalize_obs: true
  normalize_rew: true
  policy_delay: 2
  noise:
    kind: gaussian
    dim: 8
    sigma: 0.1
  lr_alpha: 0.001
  init_alpha: 0.2
  auto_tune_alpha: true

training:
  num_episodes: 2
  warmup_steps: 10
  batch_size: 10
  save_every: 1
  rollout_every: 1
  wandb: false

output_dir: outputs/test
device: cpu
```

- [ ] **Step 4: Run full test suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add configs/single_drone.yaml configs/swarm.yaml configs/test.yaml
git commit -m "config: add agent.type + SAC keys + training.wandb to all configs"
```

---

## Task 10: Extended Smoke Tests + pyproject.toml

**Files:**
- Modify: `tests/test_smoke.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace tests/test_smoke.py**

```python
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
```

- [ ] **Step 2: Run smoke tests to verify all 6 pass**

```bash
pytest tests/test_smoke.py -v
```

Expected: 6/6 pass. If any fails, read stderr from the assertion message to diagnose.

- [ ] **Step 3: Update pyproject.toml — add wandb to optional deps**

In `pyproject.toml`, find the `[project.optional-dependencies]` section and add a `wandb` group (or extend it if it already exists). The file currently has `dev`, and no `wandb` group. Add it:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=7.4",
  "flake8>=6.1",
  "black>=23.0",
]
wandb = [
  "wandb>=0.16",
]
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass (the count will be higher than the original 47).

- [ ] **Step 5: Commit**

```bash
git add tests/test_smoke.py pyproject.toml
git commit -m "test: extend smoke tests to cover all algos (td3/sac/ddpg) × (single/swarm)"
```

---

## Task 11: Update evaluate.py

**Files:**
- Modify: `scripts/evaluate.py`

`evaluate.py` currently hard-codes `TD3Agent` and calls the removed `select_action_deterministic`. Fix: use `make_agent` and `select_action(obs, deterministic=True)`. Add `--algo` flag so the right agent class is instantiated for any checkpoint.

- [ ] **Step 1: Replace scripts/evaluate.py**

```python
#!/usr/bin/env python3
"""
evaluate.py — Load a checkpoint and render an episode.

Usage
-----
  python scripts/evaluate.py --config configs/single_drone.yaml \
      --checkpoint outputs/checkpoint_final.pt
  python scripts/evaluate.py --config configs/swarm.yaml --swarm \
      --checkpoint outputs/checkpoint_final.pt --algo td3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_rl.agents.factory import make_agent
from drone_rl.envs.drone_env import DroneEnv, OBS_DIM
from drone_rl.envs.swarm_env import SwarmEnv
from drone_rl.utils.visualization import plot_coverage_heatmap, plot_trajectories


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--swarm",      action="store_true")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--algo",       default=None,
                        help="Override agent.type in config (td3 | sac | ddpg)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.algo:
        cfg["agent"]["type"] = args.algo

    env = SwarmEnv(cfg) if args.swarm else DroneEnv(cfg)
    n   = cfg["env"].get("n_drones", 1) if args.swarm else 1

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = make_agent(cfg, obs_dim, action_dim)
    agent.load(args.checkpoint)

    obs, _ = env.reset(seed=args.seed)
    done = trunc = False
    obs_dim_single = OBS_DIM

    # (n_drones, T, OBS_DIM) trajectory storage
    trajectories: List[List[np.ndarray]] = [[] for _ in range(n)]
    ep_reward = 0.0
    info: dict = {}

    while not (done or trunc):
        for i in range(n):
            trajectories[i].append(
                obs[i * obs_dim_single:(i + 1) * obs_dim_single].copy()
            )
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        ep_reward += reward

    # Append the terminal observation so the trajectory includes the final state
    if any(trajectories[0]):  # only if at least one step was taken
        for i in range(n):
            trajectories[i].append(
                obs[i * obs_dim_single:(i + 1) * obs_dim_single].copy()
            )

    print(f"Episode reward:  {ep_reward:.2f}")
    print(f"Coverage:        {info.get('coverage', 0):.2%}")

    traj_arrays = [np.array(t) for t in trajectories]

    # Guard: skip plotting if no steps were taken (degenerate episode)
    if any(len(t) == 0 for t in traj_arrays):
        print("Warning: empty trajectory — no steps were taken. Skipping plots.")
        return

    x_max = cfg["env"]["x_max"]
    y_max = cfg["env"]["y_max"]

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectories(
        traj_arrays, x_max, y_max,
        output_path=str(output_dir / "trajectories.png"),
    )

    reward_fn = env._shared_reward_fn if args.swarm else env.reward_fn

    plot_coverage_heatmap(
        reward_fn.visited, x_max, y_max,
        output_path=str(output_dir / "coverage.png"),
    )

    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "fix: evaluate.py uses agent factory and select_action(deterministic=True)"
```
