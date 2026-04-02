"""SAC agent — squashed Gaussian actor, auto-entropy tuning."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drone_rl.agents.base import AgentBase, RunningMeanStd
from drone_rl.agents.replay_buffer import ReplayBuffer


class _SACGaussianActor(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor):
        h       = self.net(x)
        mean    = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor):
        mean, log_std = self(x)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u    = dist.rsample()
        action   = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def mean_action(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self(x)
        return torch.tanh(mean)


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


class SACAgent(AgentBase):
    """SAC with twin critics and automatic entropy tuning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        init_alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden: int = 256,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        normalize_obs: bool = True,
        normalize_rew: bool = True,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.device          = torch.device(device)
        self.gamma           = gamma
        self.tau             = tau
        self.grad_clip       = grad_clip
        self.batch_size      = batch_size
        self.normalize_obs   = normalize_obs
        self.normalize_rew   = normalize_rew
        self.auto_tune_alpha = auto_tune_alpha
        self.target_entropy  = float(-action_dim)

        self.actor          = _SACGaussianActor(state_dim, action_dim, hidden).to(self.device)
        self.critic1        = _Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2        = _Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic2_target = deepcopy(self.critic2)

        self.actor_opt   = torch.optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32,
            requires_grad=True, device=self.device,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.buffer  = ReplayBuffer(buffer_capacity)
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        self.rew_rms = RunningMeanStd(shape=())

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.normalize_obs:
            obs = self.obs_rms.normalize(obs)
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.actor.mean_action(t).cpu().numpy()[0]
            else:
                a, _ = self.actor.sample(t)
                a = a.cpu().numpy()[0]
        return np.clip(a, -1.0, 1.0).astype(np.float32)

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

        # ---- Critic update ----
        with torch.no_grad():
            next_a, next_log_pi = self.actor.sample(next_s)
            target_q = rewards + self.gamma * (1 - dones) * (
                torch.min(self.critic1_target(next_s, next_a),
                          self.critic2_target(next_s, next_a))
                - self.log_alpha.exp() * next_log_pi
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

        # ---- Actor update ----
        pi, log_pi = self.actor.sample(states)
        actor_loss = (self.log_alpha.exp() * log_pi
                      - torch.min(self.critic1(states, pi),
                                  self.critic2(states, pi))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        # ---- Alpha update ----
        alpha_loss_val = 0.0
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = alpha_loss.item()

        # ---- Soft target update ----
        for tgt, src in ((self.critic1_target, self.critic1),
                          (self.critic2_target, self.critic2)):
            for tp, sp in zip(tgt.parameters(), src.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return {
            "critic_loss": sum(c_losses) / len(c_losses),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
            "alpha_loss":  alpha_loss_val,
            "mean_q":      self.critic1(states, actions).detach().mean().item(),
        }

    def save(self, path: str) -> None:
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic1":        self.critic1.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2":        self.critic2.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha":      self.log_alpha.detach().cpu(),
            "obs_rms_mean":   self.obs_rms.mean,
            "obs_rms_var":    self.obs_rms.var,
            "rew_rms_mean":   self.rew_rms.mean,
            "rew_rms_var":    self.rew_rms.var,
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic1.load_state_dict(ck["critic1"])
        self.critic1_target.load_state_dict(ck["critic1_target"])
        self.critic2.load_state_dict(ck["critic2"])
        self.critic2_target.load_state_dict(ck["critic2_target"])
        if "log_alpha" in ck:
            with torch.no_grad():
                self.log_alpha.copy_(ck["log_alpha"].to(self.device))
        if "obs_rms_mean" in ck:
            self.obs_rms.mean = ck["obs_rms_mean"]
            self.obs_rms.var  = ck["obs_rms_var"]
        if "rew_rms_mean" in ck:
            self.rew_rms.mean = ck["rew_rms_mean"]
            self.rew_rms.var  = ck["rew_rms_var"]
