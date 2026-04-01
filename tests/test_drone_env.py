from pathlib import Path

import numpy as np
import pytest
import yaml
from drone_rl.envs.drone_env import DroneEnv, OBS_DIM, ACTION_DIM


@pytest.fixture
def cfg():
    with open(Path(__file__).parent.parent / "configs" / "test.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def env(cfg):
    return DroneEnv(cfg)


def test_obs_space_shape(env):
    assert env.observation_space.shape == (OBS_DIM,)


def test_action_space_shape(env):
    assert env.action_space.shape == (ACTION_DIM,)


def test_action_space_bounds(env):
    assert env.action_space.low[0] == pytest.approx(-1.0)
    assert env.action_space.high[0] == pytest.approx(1.0)


def test_reset_returns_obs_and_dict(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(info, dict)


def test_reset_obs_dtype(env):
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32


def test_step_returns_correct_shapes(env):
    env.reset(seed=0)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    obs, reward, done, trunc, info = env.step(action)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(trunc, bool)


def test_action_clipping(env):
    env.reset(seed=0)
    # Extreme actions should not raise
    obs, _, _, _, _ = env.step(np.ones(ACTION_DIM) * 999.0)
    assert obs.shape == (OBS_DIM,)


def test_truncation_at_max_steps(env):
    env.reset(seed=42)
    done = trunc = False
    steps = 0
    while not (done or trunc):
        _, _, done, trunc, _ = env.step(np.zeros(ACTION_DIM))
        steps += 1
    # max_steps in test.yaml is 20; allow boundary termination before that
    assert steps <= 20


def test_boundary_violation_terminates(env, cfg):
    env.reset(seed=0)
    # Push drone far outside boundary
    env._x = cfg["env"]["x_max"] + 1.0
    action = np.zeros(ACTION_DIM)
    _, _, done, _, info = env.step(action)
    assert done is True
    assert info["boundary"] is True


def test_coverage_info_in_step(env):
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(ACTION_DIM))
    assert "coverage" in info
    assert 0.0 <= info["coverage"] <= 1.0


def test_z_stays_within_altitude(env):
    env.reset(seed=0)
    # Push drone upward repeatedly
    for _ in range(200):
        obs, _, done, trunc, _ = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
        if done or trunc:
            break
    z = obs[2]
    assert z <= env.max_altitude + 1e-6


def test_shared_reward_fn_not_reset_by_drone(env):
    """When reward_fn is passed externally, drone.reset() must not clear it."""
    from drone_rl.rewards.coverage import CoverageReward
    import yaml
    with open(Path(__file__).parent.parent / "configs" / "test.yaml") as f:
        cfg = yaml.safe_load(f)
    shared = CoverageReward(x_max=10.0, y_max=10.0, cell_size=2.0)
    d = DroneEnv(cfg, reward_fn=shared)
    d.reset(seed=0)
    shared.compute(0.0, 0.0)  # mark a cell on the shared grid
    assert shared.coverage_fraction > 0.0
    d.reset(seed=1)  # drone resets — shared grid must be untouched
    assert shared.coverage_fraction > 0.0


def test_reset_reproducible_with_seed(cfg):
    """Two resets with the same seed must produce identical observations."""
    env1 = DroneEnv(cfg)
    env2 = DroneEnv(cfg)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


def test_step_physics_one_step():
    """Verify velocity and position update formula for a known action."""
    import yaml
    from pathlib import Path
    with open(Path(__file__).parent.parent / "configs" / "test.yaml") as f:
        cfg = yaml.safe_load(f)
    env = DroneEnv(cfg)
    # Place drone at known state: center, zero velocity
    env._x = env._y = 0.0
    env._z = 2.0
    env._vx = env._vy = env._vz = 0.0

    # Action: pure roll = 1.0 (max lateral), no other forces
    from drone_rl.envs.drone_env import MAX_ACCEL
    action = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)

    dt = cfg["env"]["dt"]
    expected_vx = MAX_ACCEL * dt  # 4.0 * 0.05 = 0.2 m/s
    expected_x  = expected_vx * dt  # 0.2 * 0.05 = 0.01 m

    assert obs[3] == pytest.approx(expected_vx, abs=1e-5)  # vx
    assert obs[0] == pytest.approx(expected_x, abs=1e-5)   # x
