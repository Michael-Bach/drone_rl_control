"""Swarm environment: centralized controller for N drones."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drone_rl.envs.drone_env import ACTION_DIM, OBS_DIM, DroneEnv
from drone_rl.rewards.coverage import CoverageReward


class SwarmEnv(gym.Env):
    """
    Wraps N DroneEnv instances under a single centralized controller.

    Observation: (N * 7,) — concatenated drone states
    Action:      (N * 4,) — split and dispatched per drone
    Reward:      shared scalar (sum of individual rewards / N)

    All drones share one CoverageReward instance: visiting a cell that any
    drone has already visited yields no coverage bonus.

    Termination: the episode ends when ANY drone hits a boundary (any_done
    semantics). With larger swarms, premature termination becomes more likely.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        env_cfg = cfg["env"]
        rew_cfg = cfg["reward"]

        self.n = int(env_cfg["n_drones"])
        self.min_separation = float(env_cfg.get("min_separation", 2.0))

        self._shared_reward_fn = CoverageReward(
            x_max=float(env_cfg["x_max"]),
            y_max=float(env_cfg["y_max"]),
            cell_size=float(rew_cfg["cell_size"]),
            coverage_bonus=float(rew_cfg.get("coverage_bonus", 1.0)),
            revisit_penalty=float(rew_cfg.get("revisit_penalty", 0.0)),
            boundary_penalty=float(rew_cfg.get("boundary_penalty", 10.0)),
            proximity_penalty=float(rew_cfg.get("proximity_penalty", 0.0)),
        )

        self._drones: List[DroneEnv] = [
            DroneEnv(cfg, reward_fn=self._shared_reward_fn)
            for _ in range(self.n)
        ]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n * OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n * ACTION_DIM,), dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        self._shared_reward_fn.reset()
        obs_parts = []
        for i, drone in enumerate(self._drones):
            s = seed + i if seed is not None else None
            obs_i, _ = drone.reset(seed=s)
            obs_parts.append(obs_i)
        return np.concatenate(obs_parts), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        actions = action.reshape(self.n, ACTION_DIM)
        total_reward = 0.0
        any_done  = False
        any_trunc = False

        for i, drone in enumerate(self._drones):
            _, r_i, done_i, trunc_i, _ = drone.step(actions[i])
            total_reward += r_i
            any_done  = any_done  or done_i
            any_trunc = any_trunc or trunc_i

        positions = np.array([[d._x, d._y] for d in self._drones])
        total_reward += self._shared_reward_fn.proximity_reward(
            positions, self.min_separation
        )

        obs = np.concatenate([d._obs() for d in self._drones])
        shared_reward = total_reward / self.n

        return obs, float(shared_reward), any_done, any_trunc, {
            "coverage": self._shared_reward_fn.coverage_fraction,
        }
