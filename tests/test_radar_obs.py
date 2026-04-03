from pathlib import Path
import numpy as np
import pytest
import yaml
from drone_rl.envs.drone_env import DroneEnv, OBS_DIM, ACTION_DIM


def _cfg():
    with open(Path(__file__).parent.parent / "configs" / "test.yaml") as f:
        return yaml.safe_load(f)


def test_obs_dim_is_16():
    assert OBS_DIM == 16


def test_obs_shape_is_16():
    env = DroneEnv(_cfg())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (16,)


def test_observation_space_shape_is_16():
    env = DroneEnv(_cfg())
    assert env.observation_space.shape == (16,)


def test_radar_obs_default_zeros():
    env = DroneEnv(_cfg())
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    np.testing.assert_array_equal(obs[7:], np.zeros(9, dtype=np.float32))


def test_radar_obs_set_externally_appears_in_obs():
    env = DroneEnv(_cfg())
    env.reset(seed=0)
    env.radar_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    np.testing.assert_array_equal(obs[7:], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])


def test_radar_obs_cleared_on_reset():
    env = DroneEnv(_cfg())
    env.reset(seed=0)
    env.radar_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    env.reset(seed=1)
    obs, _, _, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    np.testing.assert_array_equal(obs[7:], np.zeros(9, dtype=np.float32))


def test_radar_obs_wrong_shape_raises():
    env = DroneEnv(_cfg())
    with pytest.raises(ValueError, match="shape"):
        env.radar_obs = np.zeros(4, dtype=np.float32)
