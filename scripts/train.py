#!/usr/bin/env python3
"""
train.py — Training entry point for drone RL.

Usage
-----
  python scripts/train.py --config configs/single_drone.yaml
  python scripts/train.py --config configs/swarm.yaml --swarm
  python scripts/train.py --config configs/test.yaml --max-steps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_rl.agents.td3 import TD3Agent
from drone_rl.envs.drone_env import DroneEnv
from drone_rl.envs.swarm_env import SwarmEnv


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_agent(cfg: Dict[str, Any], obs_dim: int, action_dim: int) -> TD3Agent:
    ag = cfg["agent"]
    noise_cfg = dict(ag["noise"])
    noise_cfg["dim"] = action_dim
    # Propagate dim into nested base if present
    if "base" in noise_cfg:
        noise_cfg["base"] = dict(noise_cfg["base"])
        noise_cfg["base"]["dim"] = action_dim
    return TD3Agent(
        state_dim=obs_dim,
        action_dim=action_dim,
        noise_cfg=noise_cfg,
        lr_actor=float(ag.get("lr_actor", 1e-4)),
        lr_critic=float(ag.get("lr_critic", 1e-3)),
        gamma=float(ag.get("gamma", 0.99)),
        tau=float(ag.get("tau", 0.005)),
        hidden=int(ag.get("hidden", 256)),
        buffer_capacity=int(ag.get("buffer_capacity", 100_000)),
        device=str(cfg.get("device", "cpu")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--swarm",     action="store_true")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override total env steps (for smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr  = cfg["training"]

    env = SwarmEnv(cfg) if args.swarm else DroneEnv(cfg)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent      = build_agent(cfg, obs_dim, action_dim)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup      = int(tr.get("warmup_steps", 1000))
    batch_size  = int(tr.get("batch_size", 64))
    save_every  = int(tr.get("save_every", 500))
    num_episodes = int(tr.get("num_episodes", 1000))

    total_steps = 0
    max_steps   = args.max_steps  # None = run full training

    for ep in tqdm(range(num_episodes), desc="Episodes"):
        obs, _ = env.reset()
        agent.noise.reset()
        done = trunc = False
        ep_reward = 0.0

        while not (done or trunc):
            if total_steps < warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, trunc, info = env.step(action)
            agent.store(obs, action, reward, next_obs, float(done))

            if total_steps >= warmup:
                agent.train(batch_size)

            obs = next_obs
            ep_reward += reward
            total_steps += 1

            if max_steps is not None and total_steps >= max_steps:
                return  # smoke test exit

        if (ep + 1) % save_every == 0:
            ckpt = output_dir / f"checkpoint_{ep + 1}.pt"
            agent.save(str(ckpt))
            tqdm.write(
                f"ep={ep+1}  reward={ep_reward:.2f}  "
                f"coverage={info.get('coverage', 0):.2%}  "
                f"saved={ckpt.name}"
            )

    agent.save(str(output_dir / "checkpoint_final.pt"))


if __name__ == "__main__":
    main()
