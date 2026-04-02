"""Twin Delayed DDPG (TD3) agent."""

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


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
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


# ---------------------------------------------------------------------------
# TD3 Agent
# ---------------------------------------------------------------------------

class TD3Agent(AgentBase):
    """
    Twin Delayed DDPG with optional obs/reward normalisation.

    Parameters
    ----------
    state_dim       : observation vector length
    action_dim      : action vector length
    noise_cfg       : config dict passed to make_noise()
    lr_actor        : actor learning rate
    lr_critic       : critic learning rate
    gamma           : discount factor
    tau             : Polyak averaging coefficient
    policy_delay    : actor update frequency (relative to critic updates)
    policy_noise    : std of target policy smoothing noise
    noise_clip      : clamp range for smoothing noise
    hidden          : hidden layer width
    buffer_capacity : replay buffer capacity
    batch_size      : training batch size
    normalize_obs   : enable online obs normalisation
    normalize_rew   : enable online reward normalisation
    grad_clip       : max gradient norm
    device          : torch device string
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        noise_cfg: Dict[str, Any],
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        policy_noise: float = 0.1,
        noise_clip: float = 0.3,
        hidden: int = 256,
        buffer_capacity: int = 200_000,
        batch_size: int = 64,
        normalize_obs: bool = True,
        normalize_rew: bool = True,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.device       = torch.device(device)
        self.gamma        = gamma
        self.tau          = tau
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.grad_clip    = grad_clip
        self.batch_size   = batch_size
        self.normalize_obs = normalize_obs
        self.normalize_rew = normalize_rew
        self._train_step  = 0

        self.actor          = Actor(state_dim, action_dim, hidden).to(self.device)
        self.actor_target   = deepcopy(self.actor)
        self.critic1        = Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2        = Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic2_target = deepcopy(self.critic2)

        self.actor_opt   = torch.optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.noise: BaseNoise = make_noise(noise_cfg)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        self.rew_rms = RunningMeanStd(shape=())

    # ------------------------------------------------------------------
    # AgentBase interface
    # ------------------------------------------------------------------

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
            r_np = self.rew_rms.normalize(rewards.squeeze(1).numpy(), clip=5.0)
            rewards = torch.as_tensor(r_np, dtype=torch.float32).unsqueeze(1)
        rewards = rewards.to(self.device)

        with torch.no_grad():
            noise  = (torch.randn_like(actions) * self.policy_noise
                      ).clamp(-self.noise_clip, self.noise_clip)
            next_a = (self.actor_target(next_s) + noise).clamp(-1.0, 1.0)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(
                self.critic1_target(next_s, next_a),
                self.critic2_target(next_s, next_a),
            )

        c_losses = []
        for critic, opt in ((self.critic1, self.critic1_opt),
                            (self.critic2, self.critic2_opt)):
            loss = F.smooth_l1_loss(critic(states, actions), target_q)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
            opt.step()
            c_losses.append(loss.item())

        critic_loss = sum(c_losses) / len(c_losses)
        mean_q      = self.critic1(states, actions).mean().item()
        actor_loss  = 0.0

        if self._train_step % self.policy_delay == 0:
            pa     = self.actor(states)
            a_loss = -self.critic1(states, pa).mean() + 1e-3 * (pa ** 2).mean()
            self.actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_opt.step()
            for tgt, src in ((self.actor_target, self.actor),
                             (self.critic1_target, self.critic1),
                             (self.critic2_target, self.critic2)):
                self._soft_update(tgt, src, self.tau)
            actor_loss = a_loss.item()

        self._train_step += 1
        return {"critic_loss": critic_loss, "actor_loss": actor_loss, "mean_q": mean_q}

    def save(self, path: str) -> None:
        torch.save({
            "actor":           self.actor.state_dict(),
            "actor_target":    self.actor_target.state_dict(),
            "critic1":         self.critic1.state_dict(),
            "critic1_target":  self.critic1_target.state_dict(),
            "critic2":         self.critic2.state_dict(),
            "critic2_target":  self.critic2_target.state_dict(),
            "obs_rms_mean":    self.obs_rms.mean,
            "obs_rms_var":     self.obs_rms.var,
            "rew_rms_mean":    self.rew_rms.mean,
            "rew_rms_var":     self.rew_rms.var,
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.actor_target.load_state_dict(ck["actor_target"])
        self.critic1.load_state_dict(ck["critic1"])
        self.critic2.load_state_dict(ck["critic2"])
        if "critic1_target" in ck:
            self.critic1_target.load_state_dict(ck["critic1_target"])
            self.critic2_target.load_state_dict(ck["critic2_target"])
        if "obs_rms_mean" in ck:
            self.obs_rms.mean = ck["obs_rms_mean"]
            self.obs_rms.var  = ck["obs_rms_var"]
        if "rew_rms_mean" in ck:
            self.rew_rms.mean = ck["rew_rms_mean"]
            self.rew_rms.var  = ck["rew_rms_var"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
