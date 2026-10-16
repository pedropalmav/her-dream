"""The layout control the analysis scripts depend on.

`pinned_layout_cfg` and the BFS oracle both decide *which grid* an experiment
measures. Getting them wrong does not raise — it silently produces numbers from a
grid the world model never trained on, which is the failure mode these cover.
"""

import numpy as np
import pytest
from minigrid.core.constants import DIR_TO_VEC
from omegaconf import OmegaConf

import her_dream.goals as goals
from experiments.common import bfs_plan, has_lava, pinned_layout_cfg, sample_lava, suite_of
from her_dream.envs import make_env

FORWARD = 2


def env_cfg(task="random-goal_", **overrides):
    base = dict(
        task=task,
        env_size=10,
        agent_start_pos_x=1,
        agent_start_pos_y=1,
        agent_start_dir=0,
        goal_pos_x=None,
        goal_pos_y=None,
        n_lava=3,
    )
    return OmegaConf.create({**base, **overrides})


class TestSuiteDetection:
    @pytest.mark.parametrize("task,expected", [("lava-grid_", "lava-grid"), ("random-goal_default", "random-goal")])
    def test_suite_of(self, task, expected):
        assert suite_of(env_cfg(task=task)) == expected

    def test_has_lava(self):
        assert has_lava(env_cfg(task="lava-grid_"))
        assert not has_lava(env_cfg(task="random-goal_"))


class TestGoalGridRuns:
    """The goal-grid suites keep the old behaviour: swapped for the pinnable env."""

    def test_task_is_forced_to_fixed_goal(self):
        assert pinned_layout_cfg(env_cfg(), goal_pos=(8, 1)).task == "fixed-goal_"

    def test_positions_are_written(self):
        cfg = pinned_layout_cfg(env_cfg(), goal_pos=(8, 1), agent_start_pos=(2, 3), agent_start_dir=2)
        assert (cfg.goal_pos_x, cfg.goal_pos_y) == (8, 1)
        assert (cfg.agent_start_pos_x, cfg.agent_start_pos_y) == (2, 3)
        assert cfg.agent_start_dir == 2

    def test_original_is_not_mutated(self):
        cfg = env_cfg()
        pinned_layout_cfg(cfg, goal_pos=(8, 1))
        assert cfg.task == "random-goal_"
        assert cfg.goal_pos_x is None

    def test_no_lava_key_added(self):
        assert pinned_layout_cfg(env_cfg(), goal_pos=(8, 1)).get("lava_pos") is None


class TestLavaRuns:
    """A lava run keeps its own env — its WM was trained on grids that have lava."""

    def test_task_is_preserved(self):
        cfg = pinned_layout_cfg(env_cfg(task="lava-grid_"), goal_pos=(8, 1), lava_pos=[(3, 3)])
        assert cfg.task == "lava-grid_"

    def test_lava_is_written(self):
        cfg = pinned_layout_cfg(env_cfg(task="lava-grid_"), goal_pos=(8, 1), lava_pos=[(3, 3), (4, 5)])
        assert cfg.lava_pos == [[3, 3], [4, 5]]

    def test_missing_lava_pos_raises(self):
        """Without it the lava resamples every reset and nothing is pinned at all —
        the silent-wrong-layout case, made loud."""
        with pytest.raises(ValueError, match="needs lava_pos"):
            pinned_layout_cfg(env_cfg(task="lava-grid_"), goal_pos=(8, 1))

    def test_empty_lava_is_a_valid_pin(self):
        """Distinct from None: a deliberate zero-hazard layout, not an omission."""
        cfg = pinned_layout_cfg(env_cfg(task="lava-grid_"), goal_pos=(8, 1), lava_pos=[])
        assert cfg.lava_pos == []

    def test_pinned_spawn_overrides_agent_start_random(self):
        """A run trained with agent_start_random=True must still spawn where pinned."""
        cfg = pinned_layout_cfg(env_cfg(agent_start_random=True), agent_start_pos=(2, 3))
        assert cfg.agent_start_random is False

    def test_agent_start_random_kept_when_spawn_not_pinned(self):
        cfg = pinned_layout_cfg(env_cfg(agent_start_random=True), goal_pos=(8, 1))
        assert cfg.agent_start_random is True

    def test_pinned_spawn_is_honoured_by_a_random_start_lava_env(self):
        raw = OmegaConf.to_container(env_cfg(task="lava-grid_"), resolve=True)
        raw.update(
            action_repeat=1,
            time_limit=40,
            render_mode="rgb_array",
            mission_text=False,
            agent_start_random=True,
            stochastic_classes=4,
            stochastic_rows=2,
            goal_type="full",
        )
        cfg = pinned_layout_cfg(
            OmegaConf.create(goals.with_default_descriptors(raw)),
            goal_pos=(8, 1),
            agent_start_pos=(2, 3),
            lava_pos=[(5, 5)],
        )
        env = make_env(cfg, 0)
        for _ in range(5):
            env.reset()
            assert tuple(int(v) for v in env.unwrapped.agent_pos) == (2, 3)

    def test_pinned_layout_builds_a_real_env(self):
        raw = OmegaConf.to_container(env_cfg(task="lava-grid_"), resolve=True)
        raw.update(
            action_repeat=1,
            time_limit=40,
            render_mode="rgb_array",
            mission_text=False,
            agent_start_random=False,
            stochastic_classes=4,
            stochastic_rows=2,
            goal_type="full",
        )
        cfg = pinned_layout_cfg(
            OmegaConf.create(goals.with_default_descriptors(raw)),
            goal_pos=(8, 1),
            agent_start_pos=(1, 1),
            lava_pos=[(3, 3), (4, 5)],
        )
        env = make_env(cfg, 0)
        for _ in range(5):
            env.reset()
            assert [tuple(c) for c in env.get_wrapper_attr("lava_positions")] == [(3, 3), (4, 5)]
            assert tuple(env.get_wrapper_attr("goal_position")) == (8, 1)


class TestOracleRoutesAroundLava:
    def test_plan_avoids_lava_cells(self):
        """A wall of lava across the corridor; the plan must go around, not through."""
        lava = [(3, y) for y in range(1, 8)]
        plan = bfs_plan(10, (1, 4), 0, (5, 4), None, lava=lava)
        assert plan is not None
        assert self.cells_visited(plan, (1, 4), 0).isdisjoint(set(lava))

    def test_fully_walled_target_is_unreachable(self):
        """None, rather than a plan that walks the oracle into the fire."""
        lava = [(2, 1), (2, 2), (2, 3), (1, 2)]
        assert bfs_plan(4 + 1, (1, 1), 0, (3, 3), None, lava=lava) is None

    def test_lava_free_plan_is_unchanged(self):
        assert bfs_plan(10, (1, 1), 0, (4, 1), 0) == bfs_plan(10, (1, 1), 0, (4, 1), 0, lava=[])

    @staticmethod
    def cells_visited(plan, start_pos, start_dir):
        x, y, d = *start_pos, start_dir
        seen = {(x, y)}
        for action in plan:
            if action == 0:
                d = (d - 1) % 4
            elif action == 1:
                d = (d + 1) % 4
            else:
                dx, dy = DIR_TO_VEC[d]
                x, y = x + int(dx), y + int(dy)
                seen.add((x, y))
        return seen


class TestSampleLava:
    def test_count(self):
        assert len(sample_lava(10, 4, np.random.default_rng(0))) == 4

    def test_cells_are_distinct(self):
        cells = sample_lava(10, 8, np.random.default_rng(1))
        assert len(set(cells)) == len(cells)

    def test_avoid_is_respected(self):
        avoid = [(1, 1), (8, 1), (4, 4)]
        for seed in range(20):
            cells = sample_lava(10, 5, np.random.default_rng(seed), avoid=avoid)
            assert set(cells).isdisjoint(set(avoid))
