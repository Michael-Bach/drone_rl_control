"""
Exploration noise sources for continuous action spaces.

All noise classes share a common interface:
  - noise()      → np.ndarray of shape (dim,)
  - reset()      → called at episode start
  - step()       → called at every timestep (for decay schedules)

Use make_noise(cfg: dict) to instantiate from a config dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseNoise:
    dim: int

    def __call__(self) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        """Called at the start of each episode."""

    def step(self) -> None:
        """Called every environment step (for decay schedules)."""


# ---------------------------------------------------------------------------
# Concrete noise types
# ---------------------------------------------------------------------------

class GaussianNoise(BaseNoise):
    """Independent Gaussian noise."""

    def __init__(self, dim: int, sigma: float = 0.1) -> None:
        self.dim = dim
        self.sigma = sigma

    def __call__(self) -> np.ndarray:
        return (self.sigma * np.random.randn(self.dim)).astype(np.float32)


class OUNoise(BaseNoise):
    """Ornstein–Uhlenbeck correlated noise (temporally smooth)."""

    def __init__(self, dim: int, theta: float = 0.15,
                 sigma: float = 0.2, dt: float = 1e-2) -> None:
        self.dim = dim
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self._state = np.zeros(dim, dtype=np.float32)

    def reset(self) -> None:
        self._state[:] = 0.0

    def __call__(self) -> np.ndarray:
        dx = (
            self.theta * (-self._state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.dim)
        )
        self._state += dx
        return self._state.copy()


class EpsilonGreedyNoise(BaseNoise):
    """With probability ε, return uniform random action; else zero."""

    def __init__(self, dim: int, epsilon: float = 0.3) -> None:
        self.dim = dim
        self.epsilon = epsilon

    def __call__(self) -> np.ndarray:
        if np.random.rand() < self.epsilon:
            return np.random.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)
        return np.zeros(self.dim, dtype=np.float32)


class SparseWeaponNoise(BaseNoise):
    """
    Injects random firing signals to encourage rare weapon-use events.

    For each ship, with probability p_fire, the fire_fraction dimension is
    set to a random value in [fire_low, fire_high].
    """

    def __init__(self, n_ships: int, p_fire: float = 0.05,
                 fire_low: float = 0.6, fire_high: float = 1.0) -> None:
        self.n_ships = n_ships
        self.p_fire = p_fire
        self.fire_low = fire_low
        self.fire_high = fire_high
        self.dim = 4 * n_ships   # 4 actions per ship

    def __call__(self) -> np.ndarray:
        noise = np.zeros(self.dim, dtype=np.float32)
        for k in range(self.n_ships):
            if np.random.rand() < self.p_fire:
                noise[4 * k + 2] = np.random.uniform(self.fire_low, self.fire_high)
                noise[4 * k + 3] = np.random.uniform(-1.0, 1.0)
        return noise


class CompositeNoise(BaseNoise):
    """Sum of multiple noise sources (must share the same dim)."""

    def __init__(self, sources: List[BaseNoise]) -> None:
        assert sources, "At least one noise source required"
        self.sources = sources
        self.dim = sources[0].dim

    def reset(self) -> None:
        for s in self.sources:
            s.reset()

    def step(self) -> None:
        for s in self.sources:
            s.step()

    def __call__(self) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        for s in self.sources:
            out += s()
        return out


class ExpDecayNoise(BaseNoise):
    """Wraps any noise source and decays its scale over time."""

    def __init__(
        self,
        base: BaseNoise,
        scale_init: float = 1.0,
        gamma: float = 0.999,
        scale_min: float = 0.05,
        per: str = "step",       # "step" or "episode"
    ) -> None:
        assert per in ("step", "episode")
        self.base = base
        self.scale = scale_init
        self.gamma = gamma
        self.scale_min = scale_min
        self.per = per
        self.dim = base.dim

    def reset(self) -> None:
        self.base.reset()
        if self.per == "episode":
            self.scale = max(self.scale * self.gamma, self.scale_min)

    def step(self) -> None:
        if self.per == "step":
            self.scale = max(self.scale * self.gamma, self.scale_min)

    def __call__(self) -> np.ndarray:
        out = self.base() * self.scale
        self.step()
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_noise(cfg: Dict[str, Any]) -> BaseNoise:
    """
    Build a noise object from a config dict.

    Supported kinds
    ---------------
    gaussian / gauss / normal
    ou / ornstein-uhlenbeck
    epsilon / epsilon-greedy
    sparse / sparse-weapon / weapon
    composite / combo
    decay / annealed / exp-decay

    Example cfg (YAML-equivalent)
    ------------------------------
    kind: decay
    per: episode
    gamma: 0.995
    scale_min: 0.02
    base:
      kind: epsilon-greedy
      dim: 4
      epsilon: 1.0
    """
    cfg = dict(cfg)          # shallow copy — don't mutate caller's dict
    kind = cfg.pop("kind").lower()
    dim  = cfg.pop("dim", None)

    if kind in ("gaussian", "gauss", "normal"):
        return GaussianNoise(dim=dim, sigma=cfg.get("sigma", 0.1))

    if kind in ("ou", "ornstein-uhlenbeck"):
        return OUNoise(dim=dim,
                       theta=cfg.get("theta", 0.15),
                       sigma=cfg.get("sigma", 0.2),
                       dt=cfg.get("dt", 1e-2))

    if kind in ("epsilon", "epsilon-greedy", "eps"):
        return EpsilonGreedyNoise(dim=dim, epsilon=cfg.get("epsilon", 0.3))

    if kind in ("sparse", "sparse-weapon", "weapon"):
        n_ships = cfg.pop("n_ships")
        return SparseWeaponNoise(
            n_ships=n_ships,
            p_fire=cfg.get("p_fire", 0.05),
            fire_low=cfg.get("fire_low", 0.6),
            fire_high=cfg.get("fire_high", 1.0),
        )

    if kind in ("composite", "combo"):
        sources = [make_noise(p) for p in cfg["parts"]]
        return CompositeNoise(sources)

    if kind in ("decay", "annealed", "exp-decay"):
        base_cfg = dict(cfg.pop("base"))
        if dim is not None and "dim" not in base_cfg:
            base_cfg["dim"] = dim
        base = make_noise(base_cfg)
        return ExpDecayNoise(
            base=base,
            scale_init=cfg.get("scale_init", 1.0),
            gamma=cfg.get("gamma", 0.999),
            scale_min=cfg.get("scale_min", 0.05),
            per=cfg.get("per", "step"),
        )

    raise ValueError(f"Unknown noise kind: {kind!r}")
