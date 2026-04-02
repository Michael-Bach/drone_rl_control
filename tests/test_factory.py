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


def test_make_agent_inherits_batch_size_from_training():
    agent = make_agent(_cfg("td3"), obs_dim=11, action_dim=4)
    assert agent.batch_size == 10  # from _BASE_CFG["training"]["batch_size"]
