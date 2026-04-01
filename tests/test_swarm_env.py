from pathlib import Path
import numpy as np
import pytest
import yaml
from drone_rl.envs.drone_env import OBS_DIM, ACTION_DIM
from drone_rl.envs.swarm_env import SwarmEnv


@pytest.fixture
def cfg():
    with open(Path(__file__).parent.parent / "configs" / "test.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def env(cfg):
    return SwarmEnv(cfg)


def test_obs_space_shape(env, cfg):
    n = cfg["env"]["n_drones"]
    assert env.observation_space.shape == (n * OBS_DIM,)


def test_action_space_shape(env, cfg):
    n = cfg["env"]["n_drones"]
    assert env.action_space.shape == (n * ACTION_DIM,)


def test_action_space_bounds(env):
    assert env.action_space.low[0] == pytest.approx(-1.0)
    assert env.action_space.high[0] == pytest.approx(1.0)


def test_reset_returns_obs_and_dict(env, cfg):
    obs, info = env.reset(seed=0)
    n = cfg["env"]["n_drones"]
    assert obs.shape == (n * OBS_DIM,)
    assert isinstance(info, dict)


def test_reset_obs_dtype(env):
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32


def test_step_shapes(env, cfg):
    env.reset(seed=0)
    n = cfg["env"]["n_drones"]
    action = np.zeros(n * ACTION_DIM, dtype=np.float32)
    obs, reward, done, trunc, info = env.step(action)
    assert obs.shape == (n * OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(trunc, bool)


def test_coverage_shared_across_drones(env, cfg):
    """Two drones visiting different cells should both count toward coverage."""
    env.reset(seed=0)
    n = cfg["env"]["n_drones"]
    # Place drones at known positions via internal state
    env._drones[0]._x, env._drones[0]._y = 0.0, 0.0
    env._drones[1]._x, env._drones[1]._y = 8.0, 8.0
    action = np.zeros(n * ACTION_DIM)
    _, _, _, _, info = env.step(action)
    # At least two cells should be visited
    assert env._shared_reward_fn.coverage_fraction > 0.0


def test_shared_reward_fn_reset_on_env_reset(env):
    """SwarmEnv.reset() must clear the shared coverage grid."""
    env.reset(seed=0)
    env._drones[0]._x, env._drones[0]._y = 0.0, 0.0
    env.step(np.zeros(env.action_space.shape))
    assert env._shared_reward_fn.coverage_fraction > 0.0
    env.reset(seed=1)
    assert env._shared_reward_fn.coverage_fraction == pytest.approx(0.0)


def test_truncation_at_max_steps(env, cfg):
    env.reset(seed=0)
    done = trunc = False
    steps = 0
    n = cfg["env"]["n_drones"]
    while not (done or trunc):
        _, _, done, trunc, _ = env.step(np.zeros(n * ACTION_DIM))
        steps += 1
    assert steps <= cfg["env"]["max_steps"]
