#!/usr/bin/env python3
"""
train.py — Training entry point for drone RL.

Usage
-----
  python scripts/train.py --config configs/single_drone.yaml
  python scripts/train.py --config configs/single_drone.yaml --algo sac
  python scripts/train.py --config configs/swarm.yaml --swarm --algo td3
  python scripts/train.py --config configs/test.yaml --max-steps 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_rl.agents.factory import make_agent
from drone_rl.envs.drone_env import DroneEnv
from drone_rl.envs.swarm_env import SwarmEnv
from drone_rl.utils.logger import TrainingLogger


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--swarm",     action="store_true")
    parser.add_argument("--algo",      default=None,
                        help="Override agent.type in config (td3 | sac | ddpg)")
    parser.add_argument("--wandb",     action="store_true",
                        help="Enable W&B logging (overrides config)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override total env steps (for smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.algo:
        cfg["agent"]["type"] = args.algo
    if args.wandb:
        cfg.setdefault("training", {})["wandb"] = True

    tr  = cfg["training"]
    env = SwarmEnv(cfg) if args.swarm else DroneEnv(cfg)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent      = make_agent(cfg, obs_dim, action_dim)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_name = cfg["agent"].get("type", "td3")
    run_name  = f"{algo_name}_{'swarm' if args.swarm else 'single'}"
    logger    = TrainingLogger(
        run_name=run_name,
        output_dir=str(output_dir),
        use_wandb=bool(tr.get("wandb", False)),
        cfg=cfg,
    )

    warmup       = int(tr.get("warmup_steps", 1000))
    save_every   = int(tr.get("save_every", 500))
    num_episodes = int(tr.get("num_episodes", 1000))

    total_steps = 0
    max_steps   = args.max_steps  # None = run full training

    try:
        for ep in tqdm(range(num_episodes), desc="Episodes"):
            obs, _ = env.reset()
            if hasattr(agent, "noise"):
                agent.noise.reset()
            done = trunc = False
            ep_reward    = 0.0
            ep_start     = time.time()
            ep_steps     = 0
            info: dict   = {}
            train_metrics: dict = {}

            while not (done or trunc):
                if total_steps < warmup:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs)

                next_obs, reward, done, trunc, info = env.step(action)
                agent.store(obs, action, reward, next_obs, float(done))

                if total_steps >= warmup:
                    train_metrics = agent.train_step()

                obs = next_obs
                ep_reward   += reward
                total_steps += 1
                ep_steps    += 1

                if max_steps is not None and total_steps >= max_steps:
                    return  # smoke test exit

            ep_time = time.time() - ep_start
            metrics = {
                "episode_reward":    ep_reward,
                "episode_length":    ep_steps,
                "coverage_fraction": info.get("coverage", 0.0),
                "steps_per_second":  ep_steps / max(ep_time, 1e-6),
            }
            metrics.update(train_metrics)
            logger.log(episode=ep + 1, step=total_steps, metrics=metrics)

            if (ep + 1) % save_every == 0:
                ckpt = output_dir / f"checkpoint_{ep + 1}.pt"
                agent.save(str(ckpt))
                tqdm.write(
                    f"ep={ep+1}  reward={ep_reward:.2f}  "
                    f"coverage={info.get('coverage', 0):.2%}  "
                    f"saved={ckpt.name}"
                )

        final_path = output_dir / "checkpoint_final.pt"
        agent.save(str(final_path))
        tqdm.write(f"Training complete. Final checkpoint: {final_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
