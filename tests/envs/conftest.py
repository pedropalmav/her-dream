"""Fixtures for the goal-image rendering used by goal_sample="image"."""

import pytest

from envs.fixed_goal import GoalImageGenerator

SIZE = 10  # grid size; interior cells are 1..SIZE-2


@pytest.fixture
def generator():
    return GoalImageGenerator(size=SIZE)
