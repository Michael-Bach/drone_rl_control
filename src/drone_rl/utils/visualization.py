"""Trajectory and coverage visualisation."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(
    trajectories: List[np.ndarray],
    x_max: float,
    y_max: float,
    output_path: Optional[str] = None,
    title: str = "Drone Trajectories",
) -> None:
    """
    Plot 2-D (x, y) trajectories for one or more drones.

    Parameters
    ----------
    trajectories : list of (T, 7) arrays — one per drone (columns: x,y,z,vx,vy,vz,yaw)
    x_max, y_max : patrol area half-extents (m)
    output_path  : if given, save figure to this path instead of displaying
    title        : plot title
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(trajectories), 1)))

    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i],
                alpha=0.8, label=f"Drone {i + 1}")
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", color=colors[i], s=60, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", color=colors[i], s=80, zorder=5)

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_coverage_heatmap(
    visited: np.ndarray,
    x_max: float,
    y_max: float,
    output_path: Optional[str] = None,
    title: str = "Coverage Map",
) -> None:
    """
    Render the visited grid as a binary heatmap.

    Parameters
    ----------
    visited      : (grid_w, grid_h) bool array from CoverageReward.visited
    x_max, y_max : patrol area half-extents
    output_path  : if given, save figure to this path
    title        : plot title
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        visited.T, origin="lower", cmap="Blues",
        extent=[-x_max, x_max, -y_max, y_max],
        vmin=0, vmax=1,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_trajectory_gif(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 20,
) -> None:
    """
    Save a list of RGB frames as an animated GIF.

    Parameters
    ----------
    frames      : list of (H, W, 3) uint8 arrays
    output_path : destination path (should end in .gif)
    fps         : frames per second
    """
    import imageio
    imageio.mimsave(output_path, frames, fps=fps)
