"""Abstract base class for all off-policy RL agents, plus shared utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class RunningMeanStd:
    """Online mean/variance estimation (Welford's algorithm)."""

    def __init__(self, shape: tuple, eps: float = 1e-4) -> None:
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        b_mean, b_var, b_cnt = x.mean(0), x.var(0), x.shape[0]
        delta     = b_mean - self.mean
        tot_count = self.count + b_cnt
        new_mean  = self.mean + delta * b_cnt / tot_count
        M2 = (self.var * self.count + b_var * b_cnt
              + delta ** 2 * self.count * b_cnt / tot_count)
        self.mean, self.var, self.count = new_mean, M2 / tot_count, tot_count

    def normalize(self, x: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
        x = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        if clip is not None:
            x = np.clip(x, -clip, clip)
        return x.astype(np.float32)


class AgentBase(ABC):
    """Minimal interface all off-policy agents must satisfy."""

    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return a clipped action array for the given observation."""

    @abstractmethod
    def store(self, obs: np.ndarray, action: np.ndarray, reward: float,
              next_obs: np.ndarray, done: float) -> None:
        """Store a transition and update normalisation statistics."""

    @abstractmethod
    def train_step(self) -> dict:
        """One gradient update. Returns a metric dict (empty if buffer not ready)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist agent state to file at path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore agent state from file at path."""
