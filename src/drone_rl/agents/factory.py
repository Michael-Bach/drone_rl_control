"""Agent factory: instantiate the right agent from config."""

from __future__ import annotations

from typing import Any, Dict

from drone_rl.agents.base import AgentBase


def make_agent(cfg: Dict[str, Any], obs_dim: int, action_dim: int) -> AgentBase:
    """
    Instantiate an agent from config.

    Reads cfg["agent"]["type"] (td3 | sac | ddpg).
    cfg["training"]["batch_size"] is used as the default batch size.
    """
    ag   = cfg["agent"]
    algo = str(ag.get("type", "td3")).lower()

    shared = dict(
        state_dim=obs_dim,
        action_dim=action_dim,
        lr_actor=float(ag.get("lr_actor", 3e-4)),
        lr_critic=float(ag.get("lr_critic", 3e-4)),
        gamma=float(ag.get("gamma", 0.99)),
        tau=float(ag.get("tau", 0.005)),
        hidden=int(ag.get("hidden", 256)),
        buffer_capacity=int(ag.get("buffer_capacity", 1_000_000)),
        batch_size=int(ag.get("batch_size", cfg.get("training", {}).get("batch_size", 256))),
        normalize_obs=bool(ag.get("normalize_obs", True)),
        normalize_rew=bool(ag.get("normalize_rew", True)),
        device=str(cfg.get("device", "cpu")),
    )

    def _noise_cfg() -> dict:
        noise = dict(ag["noise"])
        noise["dim"] = action_dim
        if "base" in noise:
            noise["base"] = dict(noise["base"])
            noise["base"]["dim"] = action_dim
        return noise

    if algo == "td3":
        from drone_rl.agents.td3 import TD3Agent
        return TD3Agent(
            noise_cfg=_noise_cfg(),
            policy_delay=int(ag.get("policy_delay", 2)),
            **shared,
        )

    if algo == "ddpg":
        from drone_rl.agents.ddpg import DDPGAgent
        return DDPGAgent(noise_cfg=_noise_cfg(), **shared)

    if algo == "sac":
        from drone_rl.agents.sac import SACAgent
        return SACAgent(
            lr_alpha=float(ag.get("lr_alpha", 3e-4)),
            init_alpha=float(ag.get("init_alpha", 0.2)),
            auto_tune_alpha=bool(ag.get("auto_tune_alpha", True)),
            **shared,
        )

    raise ValueError(
        f"Unknown agent type {algo!r}. Valid options: td3, sac, ddpg"
    )
