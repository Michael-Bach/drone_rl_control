import numpy as np
import pytest
from drone_rl.rewards.coverage import CoverageReward


@pytest.fixture
def reward():
    return CoverageReward(
        x_max=10.0, y_max=10.0, cell_size=2.0,
        coverage_bonus=1.0, revisit_penalty=0.5,
        boundary_penalty=10.0, proximity_penalty=5.0,
    )


def test_grid_dimensions(reward):
    # 20m wide / 2m cell = 10 cells per axis
    assert reward.grid_w == 10
    assert reward.grid_h == 10


def test_new_cell_gives_coverage_bonus(reward):
    r = reward.compute(0.0, 0.0, boundary_violated=False)
    assert r == pytest.approx(1.0)


def test_revisit_gives_penalty(reward):
    reward.compute(0.0, 0.0)  # visit once
    r = reward.compute(0.0, 0.0)  # revisit
    assert r == pytest.approx(-0.5)


def test_boundary_violation_gives_penalty(reward):
    r = reward.compute(0.0, 0.0, boundary_violated=True)
    assert r == pytest.approx(-10.0)


def test_boundary_does_not_mark_cell(reward):
    reward.compute(0.0, 0.0, boundary_violated=True)
    # Cell should not be marked as visited
    r = reward.compute(0.0, 0.0, boundary_violated=False)
    assert r == pytest.approx(1.0)


def test_reset_clears_visited(reward):
    reward.compute(0.0, 0.0)
    reward.reset()
    assert reward.coverage_fraction == pytest.approx(0.0)
    # Cell is new again after reset
    r = reward.compute(0.0, 0.0)
    assert r == pytest.approx(1.0)


def test_coverage_fraction_increments(reward):
    reward.compute(0.0, 0.0)
    reward.compute(5.0, 5.0)
    # 10x10 grid (x_max=10, cell_size=2) → 100 cells; 2 visited = 0.02
    assert reward.coverage_fraction == pytest.approx(0.02)


def test_proximity_penalty_close(reward):
    positions = np.array([[0.0, 0.0], [0.5, 0.0]])  # 0.5m apart < min_sep=1.0
    r = reward.proximity_reward(positions, min_sep=1.0)
    assert r == pytest.approx(-5.0)


def test_proximity_no_penalty_far(reward):
    positions = np.array([[0.0, 0.0], [5.0, 0.0]])  # 5m apart > min_sep=1.0
    r = reward.proximity_reward(positions, min_sep=1.0)
    assert r == pytest.approx(0.0)


def test_out_of_bounds_position_clamped(reward):
    # Positions outside the grid should be clamped to the edge cell, not raise
    r = reward.compute(999.0, 999.0, boundary_violated=False)
    assert r == pytest.approx(1.0)  # clamped to edge cell — first visit gives bonus
