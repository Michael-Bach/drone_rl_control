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
