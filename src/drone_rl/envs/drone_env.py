"""Single-drone patrol environment (Gymnasium-compatible)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drone_rl.rewards.coverage import CoverageReward

# Physical constants matching DJI Tello
MAX_SPEED    = 8.0   # m/s
MAX_ALTITUDE = 10.0  # m
MAX_ACCEL    = 4.0   # m/s²
MAX_YAW_RATE = 90.0  # deg/s

OBS_DIM    = 16  # x, y, z, vx, vy, vz, yaw, x1, y1, s1, x2, y2, s2, x3, y3, s3
ACTION_DIM = 4   # thrust, roll, pitch, yaw_rate


class DroneEnv(gym.Env):
    """
    Single drone modelled as a point-mass in a bounded 3-D area.

    Observation: [x, y, z, vx, vy, vz, yaw, x1, y1, s1, x2, y2, s2, x3, y3, s3]  (16-dim, float32)
    Action:      [thrust, roll, pitch, yaw_rate] ∈ [-1,1] (4-dim, float32)

    Parameters
    ----------
    cfg        : full config dict (keys: env, reward)
    reward_fn  : optional shared CoverageReward; if None, creates its own.
                 When passed externally, the caller owns reset() — DroneEnv
                 will not call it.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Dict[str, Any],
        reward_fn: Optional[CoverageReward] = None,
    ) -> None:
        super().__init__()
        env_cfg = cfg["env"]
        rew_cfg = cfg["reward"]

        self.x_max        = float(env_cfg["x_max"])
        self.y_max        = float(env_cfg["y_max"])
        self.max_altitude = float(env_cfg.get("max_altitude", MAX_ALTITUDE))
        self.max_speed    = float(env_cfg.get("max_speed", MAX_SPEED))
        self.dt           = float(env_cfg.get("dt", 0.05))
        self.max_steps    = int(env_cfg["max_steps"])

        self._owns_reward_fn = reward_fn is None
        self.reward_fn = reward_fn or CoverageReward(
            x_max=self.x_max,
            y_max=self.y_max,
            cell_size=float(rew_cfg["cell_size"]),
            coverage_bonus=float(rew_cfg.get("coverage_bonus", 1.0)),
            revisit_penalty=float(rew_cfg.get("revisit_penalty", 0.0)),
            boundary_penalty=float(rew_cfg.get("boundary_penalty", 10.0)),
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        # Mutable state (initialised in reset)
        self._x = self._y = self._z = 0.0
        self._vx = self._vy = self._vz = 0.0
        self._yaw = 0.0
        self._step_count = 0
        self._radar_obs: np.ndarray = np.zeros(9, dtype=np.float32)

    @property
    def radar_obs(self) -> np.ndarray:
        return self._radar_obs

    @radar_obs.setter
    def radar_obs(self, value: np.ndarray) -> None:
        v = np.asarray(value, dtype=np.float32)
        if v.shape != (9,):
            raise ValueError(f"radar_obs must have shape (9,), got {v.shape}")
        self._radar_obs = v

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._x   = float(self.np_random.uniform(-self.x_max * 0.8, self.x_max * 0.8))
        self._y   = float(self.np_random.uniform(-self.y_max * 0.8, self.y_max * 0.8))
        self._z   = float(self.np_random.uniform(0.5, self.max_altitude * 0.5))
        self._vx  = self._vy = self._vz = 0.0
        self._yaw = float(self.np_random.uniform(0.0, 360.0))
        self._step_count = 0
        self._radar_obs = np.zeros(9, dtype=np.float32)

        if self._owns_reward_fn:
            self.reward_fn.reset()

        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)
        thrust, roll, pitch, yaw_rate = action

        # Velocity update (body-frame simplified: roll→x, pitch→y, thrust→z)
        self._vx = float(np.clip(
            self._vx + roll   * MAX_ACCEL * self.dt,
            -self.max_speed, self.max_speed,
        ))
        self._vy = float(np.clip(
            self._vy + pitch  * MAX_ACCEL * self.dt,
            -self.max_speed, self.max_speed,
        ))
        self._vz = float(np.clip(
            self._vz + thrust * MAX_ACCEL * self.dt,
            -self.max_speed, self.max_speed,
        ))
        self._yaw = (self._yaw + yaw_rate * MAX_YAW_RATE * self.dt) % 360.0

        # Position update
        self._x += self._vx * self.dt
        self._y += self._vy * self.dt
        self._z  = float(np.clip(
            self._z + self._vz * self.dt, 0.0, self.max_altitude
        ))

        self._step_count += 1

        # z is clipped to [0, max_altitude]; z <= 0.0 fires on ground contact
        # (intentional: ground contact ends the episode regardless of thrust direction)
        boundary = (
            abs(self._x) > self.x_max
            or abs(self._y) > self.y_max
            or self._z <= 0.0
        )
        reward = self.reward_fn.compute(self._x, self._y, boundary_violated=boundary)
        done   = bool(boundary)
        trunc  = self._step_count >= self.max_steps

        return self._obs(), float(reward), done, trunc, {
            "coverage": self.reward_fn.coverage_fraction,
            "boundary": boundary,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            np.array(
                [self._x, self._y, self._z,
                 self._vx, self._vy, self._vz, self._yaw],
                dtype=np.float32,
            ),
            self._radar_obs,
        ])
