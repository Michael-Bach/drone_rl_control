"""
TelloEnv — drop-in replacement for DroneEnv that commands a real DJI Tello.

Swap DroneEnv for TelloEnv at deploy time. SwarmEnv and TD3Agent are untouched.

Install hardware extras first:
  pip install "drone-rl[hardware]"

Manual test procedure (requires physical Tello):
  1. Power on Tello, connect laptop to its WiFi SSID (TELLO-XXXXXX)
  2. Run:
       python -c "
       import yaml
       from drone_rl.utils.hardware import TelloEnv
       with open('configs/single_drone.yaml') as f:
           cfg = yaml.safe_load(f)
       e = TelloEnv(cfg)
       obs, _ = e.reset()
       print('obs:', obs)
       e.close()
       "
  3. Tello should take off to ~1 m, hover, then land on e.close()
  4. Confirm telemetry prints: [x, y, z, vx, vy, vz, yaw]

Radar station integration:
  Extend _read_telemetry() to query radar station TCP/serial API and append
  detections to the state vector. Update OBS_DIM in the caller accordingly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# djitellopy is an optional hardware dependency. The guard allows the rest of
# the package to import cleanly without hardware extras installed.
try:
    from djitellopy import Tello
    _TELLO_AVAILABLE = True
except ImportError:
    _TELLO_AVAILABLE = False

# Tello RC command scale: actions in [-1, 1] map to [-100, 100] integers
_RC_SCALE = 100


class TelloEnv:
    """
    Wraps a physical DJI Tello with the same step()/reset() interface as DroneEnv.

    Observation: [x, y, z, vx, vy, vz, yaw] (7-dim, float32)
                 x, y are dead-reckoned from velocity (Tello has no absolute positioning).
    Action:      [thrust, roll, pitch, yaw_rate] ∈ [-1, 1]

    The step() reward is always 0.0 — reward is computed externally using the
    same CoverageReward as the simulation. Pass the shared CoverageReward to
    SwarmEnv as usual; TelloEnv is used only as the innermost physics layer.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        if not _TELLO_AVAILABLE:
            raise ImportError(
                "djitellopy is required for TelloEnv. "
                "Install with: pip install 'drone-rl[hardware]'"
            )
        env_cfg = cfg["env"]
        self.dt           = float(env_cfg.get("dt", 0.05))
        self.max_steps    = int(env_cfg["max_steps"])
        self.max_altitude = float(env_cfg.get("max_altitude", 10.0))

        self._tello = Tello()
        self._tello.connect()
        # Video stream not opened — TelloEnv uses only telemetry (state dict).
        # Add self._tello.streamon() here if camera input is needed in future.

        # Dead-reckoned position (Tello has no absolute positioning sensor)
        self._x = self._y = self._z = 0.0
        self._vx = self._vy = self._vz = 0.0
        self._yaw = 0.0
        self._step_count = 0
        # TODO: set self.radar_obs (float32[4]) from radar station telemetry to match DroneEnv interface

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Take off to ~1 m and return initial telemetry."""
        self._x = self._y = 0.0
        self._z = 1.0
        self._vx = self._vy = self._vz = 0.0
        self._yaw = 0.0
        self._step_count = 0

        self._tello.takeoff()
        return self._read_telemetry(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Send RC commands and return updated telemetry. Reward is always 0.0."""
        action = np.clip(action, -1.0, 1.0)
        thrust, roll, pitch, yaw_rate = action

        self._tello.send_rc_control(
            left_right_velocity=int(roll      * _RC_SCALE),
            forward_backward_velocity=int(pitch  * _RC_SCALE),
            up_down_velocity=int(thrust           * _RC_SCALE),
            yaw_velocity=int(yaw_rate             * _RC_SCALE),
        )

        obs = self._read_telemetry()
        self._step_count += 1
        trunc = self._step_count >= self.max_steps

        # Reward is always 0.0 — computed externally by CoverageReward
        return obs, 0.0, False, trunc, {}

    def close(self) -> None:
        """Land and disconnect."""
        self._tello.land()
        self._tello.end()

    def _read_telemetry(self) -> np.ndarray:
        """
        Read state from Tello SDK and dead-reckon position.

        Tello velocity is in cm/s; converted to m/s here.
        Height is in cm; converted to m.
        Radar station detections can be added here as extra state dimensions.
        """
        # Read the full state dict once; .get(key, 0) is safe when telemetry
        # is not yet available (avoids TelloException from individual getters).
        state   = self._tello.get_current_state()
        vx_cms  = state.get("vgx", 0)
        vy_cms  = state.get("vgy", 0)
        vz_cms  = state.get("vgz", 0)
        yaw_deg = state.get("yaw", 0)
        z_cm    = state.get("h", 0)

        self._vx = vx_cms / 100.0
        self._vy = vy_cms / 100.0
        self._vz = vz_cms / 100.0
        # x, y: dead-reckoned by integrating velocity (no GPS on Tello)
        self._x  += self._vx * self.dt
        self._y  += self._vy * self.dt
        # z: taken directly from barometric/ToF height sensor — more accurate
        # than integrating vz. Note: _z and _vz are kinematically inconsistent
        # (z does not equal z_prev + vz*dt), which is a known sim-to-real gap.
        self._z   = float(z_cm) / 100.0
        self._yaw = float(yaw_deg) % 360.0

        # TODO: set self.radar_obs (float32[4]) from radar station telemetry to match DroneEnv interface

        return np.array(
            [self._x, self._y, self._z,
             self._vx, self._vy, self._vz, self._yaw],
            dtype=np.float32,
        )
