"""Coverage reward for drone patrol tasks."""

from __future__ import annotations

import numpy as np


class CoverageReward:
    """
    Tracks which grid cells have been visited and returns shaped rewards.

    Parameters
    ----------
    x_max, y_max       : half-extents of the patrol area (m)
    cell_size          : grid resolution (m)
    coverage_bonus     : reward per newly visited cell
    revisit_penalty    : penalty for returning to an already-visited cell
    boundary_penalty   : penalty on boundary or altitude violation
    proximity_penalty  : penalty when any drone pair is closer than min_sep
    """

    def __init__(
        self,
        x_max: float,
        y_max: float,
        cell_size: float,
        coverage_bonus: float = 1.0,
        revisit_penalty: float = 0.0,
        boundary_penalty: float = 10.0,
        proximity_penalty: float = 0.0,
    ) -> None:
        self.x_max = x_max
        self.y_max = y_max
        self.cell_size = cell_size
        self.coverage_bonus = coverage_bonus
        self.revisit_penalty = revisit_penalty
        self.boundary_penalty = boundary_penalty
        self.proximity_penalty = proximity_penalty

        self.grid_w = int(2 * x_max / cell_size)
        self.grid_h = int(2 * y_max / cell_size)
        self.visited = np.zeros((self.grid_w, self.grid_h), dtype=bool)

    def reset(self) -> None:
        self.visited[:] = False

    def compute(self, x: float, y: float, boundary_violated: bool = False) -> float:
        """Reward for a single position update."""
        if boundary_violated:
            return -self.boundary_penalty
        gx, gy = self._cell(x, y)
        if not self.visited[gx, gy]:
            self.visited[gx, gy] = True
            return self.coverage_bonus
        return -self.revisit_penalty

    def proximity_reward(self, positions: np.ndarray, min_sep: float) -> float:
        """
        Returns -proximity_penalty if any drone pair is closer than min_sep, else 0.
        positions: (N, 2) array of (x, y) positions.
        """
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(positions[i] - positions[j]) < min_sep:
                    return -self.proximity_penalty
        return 0.0

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        gx = int((x + self.x_max) / self.cell_size)
        gy = int((y + self.y_max) / self.cell_size)
        return (
            int(np.clip(gx, 0, self.grid_w - 1)),
            int(np.clip(gy, 0, self.grid_h - 1)),
        )

    @property
    def coverage_fraction(self) -> float:
        return float(self.visited.mean())
