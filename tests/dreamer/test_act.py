"""Tests for inference-time helpers: `act`, `get_initial_state`, `_random_action`.

`act` runs the frozen WM forward pass and either samples the actor or (with
`random=True`) bypasses it. It returns `(action, state_td, extra)` and, for the
distribution-based goal types, stashes the posterior `logit` in the state.
`_random_action` produces env-valid actions for each action-space kind.
"""

import pytest
import torch

from tests.dreamer.conftest import ACT, RDETER, RK, RS, make_default_obs, make_real_dreamer


def is_one_hot(z):
    return torch.all(z.sum(dim=-1) == 1) and torch.all((z == 0) | (z == 1))


class TestAct:
    def test_returns_action_state_extra(self, default_dreamer, default_obs):
        state = default_dreamer.get_initial_state(2)
        action, new_state, extra = default_dreamer.act(default_obs, state)
        assert action.shape == (2, ACT)
        assert set(new_state.keys()) >= {"stoch", "deter", "prev_action"}
        assert "obs_step_sample_log_prob" in extra

    def test_eval_uses_mode_and_runs(self, default_dreamer, default_obs):
        state = default_dreamer.get_initial_state(2)
        action, _, _ = default_dreamer.act(default_obs, state, eval=True)
        assert action.shape == (2, ACT)

    def test_random_bypasses_actor(self, default_dreamer, default_obs):
        state = default_dreamer.get_initial_state(2)
        action, _, _ = default_dreamer.act(default_obs, state, random=True)
        # random discrete action is a valid one-hot.
        assert is_one_hot(action)

    @pytest.mark.parametrize("goal_type", ["argmax_full", "log_prob", "prob"])
    def test_distribution_goal_types_stash_logit(self, goal_type):
        # These goal types keep a (RS, RK) goal, so the default obs fits.
        agent, _ = make_real_dreamer(goal_type=goal_type)
        state = agent.get_initial_state(2)
        _, new_state, _ = agent.act(make_default_obs(), state)
        assert "logit" in new_state

    def test_plain_goal_type_has_no_logit(self, default_dreamer, default_obs):
        state = default_dreamer.get_initial_state(2)
        _, new_state, _ = default_dreamer.act(default_obs, state)
        assert "logit" not in new_state

    def test_action_does_not_require_grad(self, default_dreamer, default_obs):
        # act is @torch.no_grad.
        state = default_dreamer.get_initial_state(2)
        action, _, _ = default_dreamer.act(default_obs, state)
        assert not action.requires_grad


class TestGetInitialState:
    def test_shapes(self, default_dreamer):
        state = default_dreamer.get_initial_state(3)
        assert state["stoch"].shape == (3, RS, RK)
        assert state["deter"].shape == (3, RDETER)
        assert state["prev_action"].shape == (3, ACT)

    def test_prev_action_is_zero(self, default_dreamer):
        state = default_dreamer.get_initial_state(3)
        assert torch.all(state["prev_action"] == 0)


class TestRandomAction:
    def test_discrete_is_one_hot(self):
        agent, _ = make_real_dreamer(act="discrete")
        action = agent._random_action(4)
        assert action.shape == (4, ACT)
        assert is_one_hot(action)

    def test_multi_discrete_concatenated_one_hots(self):
        agent, _ = make_real_dreamer(act="multi")
        action = agent._random_action(4)
        assert action.shape == (4, 5)  # sum of nvec (2, 3)
        # each per-group slice is a one-hot
        assert is_one_hot(action[:, :2])
        assert is_one_hot(action[:, 2:])

    def test_continuous_in_unit_range(self):
        agent, _ = make_real_dreamer(act="cont")
        action = agent._random_action(4)
        assert action.shape == (4, ACT)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


class TestPlan2ExploreActs:
    def test_acts_with_the_explorer_ignoring_the_goal(self):
        # Data collection during pretraining is goal-agnostic, so the same
        # observation must yield the same action whatever goal is attached.
        agent, goal_shape = make_real_dreamer(plan2explore=True)
        obs = make_default_obs(B=2, goal_shape=goal_shape)
        state = agent.get_initial_state(2)

        def act_with(goal):
            # `preprocess` rescales obs["image"] in place, so each call gets a
            # copy; and obs_step rsamples the posterior, so each gets the seed.
            torch.manual_seed(0)
            fresh = {k: v.clone() for k, v in obs.items()}
            fresh["goal"] = goal
            return agent.act(fresh, state, eval=True)[0]

        other_goal = torch.zeros_like(obs["goal"])
        other_goal[..., -1] = 1.0
        assert torch.equal(act_with(obs["goal"].clone()), act_with(other_goal))
