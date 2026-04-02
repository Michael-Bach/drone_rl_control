"""AgentBase contract tests: all three agents must satisfy the interface."""
import numpy as np
import pytest
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


def test_sac_train_step_includes_alpha_keys():
    agent = _make_sac()
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    for _ in range(50):
        agent.store(obs, np.zeros(ACTION_DIM), 1.0, obs, 0.0)
    result = agent.train_step()
    assert "alpha" in result
    assert "alpha_loss" in result
