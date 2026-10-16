"""The lava half of the success-rate benchmark.

The suite has to stay a pure function of its seeds, ask a lava run the same
navigation questions the lava-free suite asks, and never hand it an impossible or
fatal task. On the env side, the drive's death count and the goal image both rely
on how cookie-env's LavaGrid behaves, so that behaviour is pinned here too.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

import her_dream.goals as goals
from experiments.common import bfs_plan, generate_lava_pairs, generate_pairs, make_suite, pinned_layout_cfg
from experiments.success_rate import layout_mismatch, suite_for
from her_dream.envs import make_env

SIZE, N_LAVA, N = 10, 3, 200


@pytest.fixture(scope="module")
def lava_pairs():
    return generate_lava_pairs(SIZE, N_LAVA, n=N)


@pytest.fixture(scope="module")
def plain_pairs():
    return generate_pairs(SIZE, n=N)


class TestLavaSuite:
    def test_is_deterministic(self, lava_pairs):
        assert generate_lava_pairs(SIZE, N_LAVA, n=N) == lava_pairs

    def test_same_tasks_as_random_goal(self, lava_pairs, plain_pairs):
        """Pair i is the random_goal suite's pair i plus lava, nothing else moved."""
        for lava, plain in zip(lava_pairs, plain_pairs, strict=True):
            for key in ("index", "spawn", "goal", "square", "on_square", "block"):
                assert lava[key] == plain[key]
            assert plain["lava"] == []

    def test_lava_count_and_placement(self, lava_pairs):
        for pair in lava_pairs:
            cells = {tuple(c) for c in pair["lava"]}
            assert len(cells) == N_LAVA
            occupied = {tuple(pair["spawn"][0]), tuple(pair["goal"][0]), tuple(pair["square"])}
            assert cells.isdisjoint(occupied)

    def test_goal_reachable_around_lava(self, lava_pairs, plain_pairs):
        for lava, plain in zip(lava_pairs, plain_pairs, strict=True):
            (spawn_pos, spawn_dir), (goal_pos, goal_dir) = lava["spawn"], lava["goal"]
            plan = bfs_plan(SIZE, spawn_pos, spawn_dir, goal_pos, None, lava=lava["lava"])
            assert plan is not None and len(plan) == lava["plan_len"]
            # The detour can only be longer, so the >= 4 filter still holds.
            assert lava["plan_len"] >= plain["plan_len"] >= 4
            assert bfs_plan(SIZE, spawn_pos, spawn_dir, goal_pos, goal_dir, lava=lava["lava"]) is not None

    def test_make_suite_needs_n_lava(self):
        with pytest.raises(ValueError):
            make_suite("lava_grid", SIZE, n=10)
        pairs, meta = make_suite("lava_grid", SIZE, n_lava=1, n=10)
        assert len(pairs) == 10 and meta["n_lava"] == 1 and meta["name"] == "lava_grid"


class TestSuiteRouting:
    def test_lava_run_gets_the_lava_suite(self):
        assert suite_for(OmegaConf.create({"task": "lava-grid_"})) == "lava_grid"
        assert suite_for(OmegaConf.create({"task": "random-goal_"})) == "random_goal"
        assert suite_for(OmegaConf.create({"task": "fixed-goal_"})) == "fixed_goal"

    @pytest.mark.parametrize("run_lava,suite_lava", [(1, 0), (0, 1), (2, 1)])
    def test_lava_mismatch_is_refused(self, run_lava, suite_lava):
        layout = {"size": SIZE, "spawn": ((1, 1), 0), "square": None, "n_lava": run_lava}
        suite = {"size": SIZE, "n_lava": suite_lava} if suite_lava else {"size": SIZE}
        assert layout_mismatch(layout, suite) is not None

    def test_matching_lava_run_is_scored(self):
        layout = {"size": SIZE, "spawn": ((1, 1), 0), "square": None, "n_lava": 1}
        assert layout_mismatch(layout, {"size": SIZE, "n_lava": 1}) is None


def lava_env(lava, agent_start_pos=(1, 1), agent_start_dir=0, goal_pos=(8, 1)):
    raw = dict(
        task="lava-grid_",
        env_size=SIZE,
        agent_start_pos_x=1,
        agent_start_pos_y=1,
        agent_start_dir=0,
        goal_pos_x=None,
        goal_pos_y=None,
        n_lava=len(lava),
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
        goal_pos=goal_pos,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        lava_pos=lava,
    )
    return make_env(cfg, 0)


def lava_pixels(image):
    """MiniGrid paints lava orange (255, 128, 0)."""
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return int(((r > 200) & (g > 90) & (g < 170) & (b < 60)).sum())


class TestLavaEnvContract:
    def test_death_leaves_the_agent_on_the_lava_cell(self):
        """`drive` counts a death as `done` with the agent standing on lava."""
        env = lava_env([(2, 1)])
        env.reset()
        forward = np.eye(env.action_space.shape[0], dtype=np.float32)[2]
        _obs, _reward, done, _info = env.step(forward)
        assert done
        assert tuple(int(v) for v in env.unwrapped.agent_pos) == (2, 1)

    def test_goal_image_needs_a_reset_to_show_lava(self):
        """Why success_rate / goal_observation_eval reset before encoding the goal."""
        env = lava_env([(4, 4)])
        render = env.get_wrapper_attr("goal_observation")
        before = render(agent_pos=(6, 3), agent_dir=0)["image"]
        env.reset()
        after = render(agent_pos=(6, 3), agent_dir=0)["image"]
        assert lava_pixels(before) == 0
        assert lava_pixels(after) > 0
