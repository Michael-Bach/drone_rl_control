#!/usr/bin/env python3
"""
evaluate.py — Load a checkpoint and render an episode.

Usage
-----
  python scripts/evaluate.py --config configs/single_drone.yaml \
      --checkpoint outputs/checkpoint_final.pt
  python scripts/evaluate.py --config configs/swarm.yaml --swarm \
      --checkpoint outputs/checkpoint_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_rl.agents.td3 import TD3Agent
from drone_rl.envs.drone_env import DroneEnv
from drone_rl.envs.swarm_env import SwarmEnv
from drone_rl.utils.visualization import plot_coverage_heatmap, plot_trajectories


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--swarm",      action="store_true")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = SwarmEnv(cfg) if args.swarm else DroneEnv(cfg)
    n   = cfg["env"].get("n_drones", 1) if args.swarm else 1

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ag = cfg["agent"]
    agent = TD3Agent(
        state_dim=obs_dim, action_dim=action_dim,
        noise_cfg={"kind": "gaussian", "dim": action_dim, "sigma": 0.0},
        hidden=int(ag.get("hidden", 256)),
        device=str(cfg.get("device", "cpu")),
    )
    agent.load(args.checkpoint)

    obs, _ = env.reset(seed=args.seed)
    done = trunc = False
    obs_dim_single = 7

    # (n_drones, T, 7) trajectory storage
    trajectories: List[List[np.ndarray]] = [[] for _ in range(n)]
    ep_reward = 0.0
    info: dict = {}

    while not (done or trunc):
        for i in range(n):
            trajectories[i].append(
                obs[i * obs_dim_single:(i + 1) * obs_dim_single].copy()
            )
        action = agent.select_action_deterministic(obs)
        obs, reward, done, trunc, info = env.step(action)
        ep_reward += reward

    print(f"Episode reward:  {ep_reward:.2f}")
    print(f"Coverage:        {info.get('coverage', 0):.2%}")

    traj_arrays = [np.array(t) for t in trajectories]
    x_max = cfg["env"]["x_max"]
    y_max = cfg["env"]["y_max"]

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectories(
        traj_arrays, x_max, y_max,
        output_path=str(output_dir / "trajectories.png"),
    )

    reward_fn = env._shared_reward_fn if args.swarm else env.reward_fn

    plot_coverage_heatmap(
        reward_fn.visited, x_max, y_max,
        output_path=str(output_dir / "coverage.png"),
    )

    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
