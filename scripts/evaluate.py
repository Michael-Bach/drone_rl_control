#!/usr/bin/env python3
"""
evaluate.py — Load a checkpoint and render an episode.

Usage
-----
  python scripts/evaluate.py --config configs/single_drone.yaml \
      --checkpoint outputs/checkpoint_final.pt
  python scripts/evaluate.py --config configs/swarm.yaml --swarm \
      --checkpoint outputs/checkpoint_final.pt --algo td3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_rl.agents.factory import make_agent
from drone_rl.envs.drone_env import DroneEnv, OBS_DIM
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
    parser.add_argument("--algo",       default=None,
                        help="Override agent.type in config (td3 | sac | ddpg)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.algo:
        cfg["agent"]["type"] = args.algo

    env = SwarmEnv(cfg) if args.swarm else DroneEnv(cfg)
    n   = cfg["env"].get("n_drones", 1) if args.swarm else 1

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = make_agent(cfg, obs_dim, action_dim)
    agent.load(args.checkpoint)

    obs, _ = env.reset(seed=args.seed)
    done = trunc = False
    obs_dim_single = OBS_DIM

    # (n_drones, T, OBS_DIM) trajectory storage
    trajectories: List[List[np.ndarray]] = [[] for _ in range(n)]
    ep_reward = 0.0
    info: dict = {}

    while not (done or trunc):
        for i in range(n):
            trajectories[i].append(
                obs[i * obs_dim_single:(i + 1) * obs_dim_single].copy()
            )
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        ep_reward += reward

    # Append the terminal observation so the trajectory includes the final state
    if any(trajectories[0]):  # only if at least one step was taken
        for i in range(n):
            trajectories[i].append(
                obs[i * obs_dim_single:(i + 1) * obs_dim_single].copy()
            )

    print(f"Episode reward:  {ep_reward:.2f}")
    print(f"Coverage:        {info.get('coverage', 0):.2%}")

    traj_arrays = [np.array(t) for t in trajectories]

    # Guard: skip plotting if no steps were taken (degenerate episode)
    if any(len(t) == 0 for t in traj_arrays):
        print("Warning: empty trajectory — no steps were taken. Skipping plots.")
        return

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
