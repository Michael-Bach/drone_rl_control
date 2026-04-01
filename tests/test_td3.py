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
        hidden=64, buffer_capacity=200, device="cpu",
    )


def test_select_action_shape(agent):
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    action = agent.select_action(obs)
    assert action.shape == (ACTION_DIM,)


def test_select_action_clipped(agent):
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    action = agent.select_action(obs)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_train_returns_false_when_buffer_empty(agent):
    result = agent.train(batch_size=10)
    assert result is False


def test_train_after_filling_buffer(agent):
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    for _ in range(50):
        agent.store(obs, np.zeros(ACTION_DIM), 1.0, obs, False)
    result = agent.train(batch_size=10)
    # train() returns bool; True = actor updated this step, False = critic-only step
    assert isinstance(result, bool)


def test_save_and_load(agent, tmp_path):
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    for _ in range(20):
        agent.store(obs, np.zeros(ACTION_DIM), 1.0, obs, False)
    agent.train(batch_size=10)
    path = str(tmp_path / "checkpoint.pt")
    agent.save(path)
    agent.load(path)
    action = agent.select_action_deterministic(obs)
    assert action.shape == (ACTION_DIM,)
