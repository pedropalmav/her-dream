import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from her_dream.rewards import (
    argmax_full_reward,
    first_row_reward,
    full_goal_reward,
    log_prob_reward,
    make_reward,
    max_cosine_reward,
    prob_reward,
    reward_offset,
    row_by_row_reward,
)

from .conftest import B, K, S, T, make_dist_mock


class TestMakeReward:
    """`make_reward` binds the goal type's reward offset, so it returns a closure
    rather than the bare function; these assert behaviour instead of identity."""

    def _onehot(self, idx, n=K):
        return torch.nn.functional.one_hot(torch.tensor(idx), n).float()

    def test_first_row_type_matches_bare_function(self):
        config = SimpleNamespace(goal_type="first_row")
        state = self._onehot([[0] * S] * B)
        goal = self._onehot([0] * B)
        assert torch.equal(make_reward(config)(state, goal), first_row_reward(state, goal))

    def test_row_by_row_type_matches_bare_function(self):
        config = SimpleNamespace(goal_type="row_by_row")
        state = self._onehot([[0] * S] * B)
        goal = self._onehot([0] * S)
        assert torch.equal(make_reward(config)(state, goal), row_by_row_reward(state, goal))

    def test_full_type_matches_bare_function(self):
        config = SimpleNamespace(goal_type="full")
        state = self._onehot([[0] * S] * B)
        goal = self._onehot([0] * S)
        assert torch.equal(make_reward(config)(state, goal), full_goal_reward(state, goal))

    def test_argmax_full_type_matches_bare_function(self):
        config = SimpleNamespace(goal_type="argmax_full")
        logit = torch.randn(B, S, K)
        goal = self._onehot([0] * S)
        assert torch.equal(make_reward(config)(logit, goal), argmax_full_reward(logit, goal))

    def test_log_prob_type_callable(self):
        prob_threshold = 0.5
        config = SimpleNamespace(goal_type="log_prob", prob_threshold=prob_threshold)
        fn = make_reward(config)
        assert callable(fn)
        logits = torch.randn(B, S, K)
        log_prob_val = torch.full((B,), math.log(prob_threshold))
        dist = make_dist_mock(logits, log_prob_val)
        goal = torch.zeros(S, K)
        result = fn(dist, goal)
        assert torch.all(result == 0)

    def test_prob_type_matches_bare_function(self):
        config = SimpleNamespace(goal_type="prob")
        dist = make_dist_mock(torch.randn(B, S, K), torch.log(torch.full((B,), 0.25)))
        goal = torch.zeros(S, K)
        assert torch.allclose(make_reward(config)(dist, goal), prob_reward(dist, goal))

    def test_max_cosine_type_matches_bare_function(self):
        config = SimpleNamespace(goal_type="max_cosine")
        state = torch.randn(B, S, K)
        goal = torch.randn(S, K)
        assert torch.allclose(make_reward(config)(state, goal), max_cosine_reward(state, goal))

    def test_invalid_type_raises(self):
        config = SimpleNamespace(goal_type="unknown")
        with pytest.raises(ValueError):
            make_reward(config)


class TestRewardOffset:
    """The offset is the per-step baseline: reward == offset + match."""

    def _onehot(self, idx, n=K):
        return torch.nn.functional.one_hot(torch.tensor(idx), n).float()

    @pytest.mark.parametrize(
        "goal_type,expected",
        [
            ("first_row", -1.0),
            ("row_by_row", -1.0),
            ("full", -1.0),
            ("argmax_full", -1.0),
            ("log_prob", -1.0),
            # Already non-negative / centred, so their historical baseline is 0.
            ("prob", 0.0),
            ("max_cosine", 0.0),
        ],
    )
    def test_default_backfilled_from_goal_type_config(self, goal_type, expected):
        """Configs saved before this key existed must still resolve."""
        assert reward_offset(SimpleNamespace(goal_type=goal_type)) == expected

    def test_explicit_value_wins_over_default(self):
        assert reward_offset(SimpleNamespace(goal_type="full", reward_offset=0.0)) == 0.0

    def test_offset_zero_makes_match_rewards_non_negative(self):
        """The whole point: with offset 0 nothing is ever negative, so ending an
        episode can never beat surviving it."""
        matching = self._onehot([[0] * S] * B)
        goal = self._onehot([0] * S)
        missing = self._onehot([[1] * S] * B)
        for state in (matching, missing):
            assert torch.all(full_goal_reward(state, goal, offset=0.0) >= 0)
            assert torch.all(row_by_row_reward(state, goal, offset=0.0) >= 0)

    def test_offset_shifts_by_a_constant(self):
        """Changing the offset adds a constant, it does not reshape the reward."""
        state = self._onehot([[0, 0, 1, 1][:S]] * B)
        goal = self._onehot([0] * S)
        a = row_by_row_reward(state, goal, offset=-1.0)
        b = row_by_row_reward(state, goal, offset=0.0)
        assert torch.allclose(b - a, torch.ones_like(a))

    def test_matching_state_is_offset_plus_one(self):
        state = self._onehot([[0] * S] * B)
        goal = self._onehot([0] * S)
        for off in (-1.0, 0.0, 2.5):
            assert torch.allclose(full_goal_reward(state, goal, offset=off), torch.tensor(off + 1.0))

    def test_unmatched_state_is_exactly_the_offset(self):
        state = self._onehot([[1] * S] * B)
        goal = self._onehot([0] * S)
        for off in (-1.0, 0.0, 2.5):
            assert torch.allclose(full_goal_reward(state, goal, offset=off), torch.tensor(off))


class TestFirstRowReward:
    def test_3d_match(self):
        goal = F.one_hot(torch.tensor(0), K).float()
        state = torch.zeros(B, S, K)
        state[:, 0, :] = goal
        result = first_row_reward(state, goal)
        assert result.shape == (B, 1)
        assert torch.all(result == 0)

    def test_3d_no_match(self):
        goal = F.one_hot(torch.tensor(0), K).float()
        state = F.one_hot(torch.ones(B, S, dtype=torch.long), K).float()
        result = first_row_reward(state, goal)
        assert torch.all(result == -1)

    def test_4d_match(self):
        goal = F.one_hot(torch.zeros(B, dtype=torch.long), K).float()
        state = torch.zeros(B, T, S, K)
        state[:, :, 0, :] = goal.unsqueeze(1)
        result = first_row_reward(state, goal)
        assert result.shape == (B, T, 1)
        assert torch.all(result == 0)

    def test_4d_no_match(self):
        goal = F.one_hot(torch.zeros(B, dtype=torch.long), K).float()
        state = F.one_hot(torch.ones(B, T, S, dtype=torch.long), K).float()
        result = first_row_reward(state, goal)
        assert torch.all(result == -1)

    def test_4d_output_shape(self):
        goal = F.one_hot(torch.zeros(B, dtype=torch.long), K).float()
        state = F.one_hot(torch.zeros(B, T, S, dtype=torch.long), K).float()
        assert first_row_reward(state, goal).shape == (B, T, 1)

    def test_invalid_dim_raises(self):
        state = torch.zeros(B, K)
        goal = torch.zeros(K)
        with pytest.raises(ValueError):
            first_row_reward(state, goal)


class TestRowByRowReward:
    def test_3d_all_match(self):
        goal = F.one_hot(torch.zeros(S, dtype=torch.long), K).float()
        state = goal.unsqueeze(0).expand(B, S, K).clone()
        result = row_by_row_reward(state, goal)
        assert result.shape == (B, 1)
        assert torch.all(result == 0)

    def test_3d_partial_match(self):
        goal = F.one_hot(torch.zeros(S, dtype=torch.long), K).float()
        state = F.one_hot(torch.ones(B, S, dtype=torch.long), K).float()
        state[:, 0, :] = goal[0]
        result = row_by_row_reward(state, goal)
        expected = torch.full((B, 1), 1.0 / S - 1.0)
        assert torch.allclose(result.float(), expected)

    def test_3d_no_match(self):
        goal = F.one_hot(torch.zeros(S, dtype=torch.long), K).float()
        state = F.one_hot(torch.ones(B, S, dtype=torch.long), K).float()
        result = row_by_row_reward(state, goal)
        assert torch.allclose(result.float(), torch.full((B, 1), -1.0))

    def test_4d_all_match(self):
        goal = F.one_hot(torch.zeros(B, S, dtype=torch.long), K).float()
        state = goal.unsqueeze(1).expand(B, T, S, K).clone()
        result = row_by_row_reward(state, goal)
        assert result.shape == (B, T, 1)
        assert torch.all(result == 0)

    def test_4d_output_shape(self):
        goal = F.one_hot(torch.zeros(B, S, dtype=torch.long), K).float()
        state = goal.unsqueeze(1).expand(B, T, S, K).clone()
        assert row_by_row_reward(state, goal).shape == (B, T, 1)

    def test_invalid_dim_raises(self):
        state = torch.zeros(B, K)
        goal = torch.zeros(K)
        with pytest.raises(ValueError):
            row_by_row_reward(state, goal)


class TestFullGoalReward:
    def test_3d_match(self):
        goal = F.one_hot(torch.zeros(S, dtype=torch.long), K).float()
        state = goal.unsqueeze(0).expand(B, S, K).clone()
        result = full_goal_reward(state, goal)
        assert result.shape == (B, 1)
        assert torch.all(result == 0)

    def test_3d_no_match(self):
        goal = F.one_hot(torch.zeros(S, dtype=torch.long), K).float()
        state = F.one_hot(torch.ones(B, S, dtype=torch.long), K).float()
        result = full_goal_reward(state, goal)
        assert torch.all(result == -1)

    def test_4d_match(self):
        goal = F.one_hot(torch.zeros(B, S, dtype=torch.long), K).float()
        state = goal.unsqueeze(1).expand(B, T, S, K).clone()
        result = full_goal_reward(state, goal)
        assert result.shape == (B, T, 1)
        assert torch.all(result == 0)

    def test_4d_no_match(self):
        goal = F.one_hot(torch.zeros(B, S, dtype=torch.long), K).float()
        state = F.one_hot(torch.ones(B, T, S, dtype=torch.long), K).float()
        result = full_goal_reward(state, goal)
        assert torch.all(result == -1)

    def test_4d_output_shape(self):
        goal = F.one_hot(torch.zeros(B, S, dtype=torch.long), K).float()
        state = goal.unsqueeze(1).expand(B, T, S, K).clone()
        assert full_goal_reward(state, goal).shape == (B, T, 1)

    def test_invalid_dim_raises(self):
        state = torch.zeros(B, K)
        goal = torch.zeros(K)
        with pytest.raises(ValueError):
            full_goal_reward(state, goal)


class TestOneHotComparison:
    """The exact-match rewards compare argmax indices, not float `==`, so a
    one-hot whose '1' entry is slightly off (e.g. 0.9995 under fp16) still
    matches; a non-one-hot input trips the validation assert."""

    def test_perturbed_onehot_still_matches(self):
        goal = F.one_hot(torch.zeros(B, S, dtype=torch.long), K).float()
        state = goal.clone()
        state[state == 1.0] = 0.9995  # within ONEHOT_ATOL of 1.0, like fp16
        result = full_goal_reward(state, goal)
        assert torch.all(result == 0)  # would be -1 under exact `==`

    def test_non_onehot_input_asserts(self):
        goal = F.one_hot(torch.zeros(S, dtype=torch.long), K).float()
        state = torch.zeros(B, S, K)  # degenerate: not one-hot
        with pytest.raises(AssertionError):
            row_by_row_reward(state, goal)


class TestLogProbReward:
    def test_3d_log_prob_above_threshold(self):
        logits = torch.randn(B, S, K)
        dist = make_dist_mock(logits, torch.full((B,), 1.0))
        result = log_prob_reward(dist, torch.zeros(S, K), log_threshold=0.0)
        assert torch.all(result == 0)

    def test_3d_log_prob_below_threshold(self):
        logits = torch.randn(B, S, K)
        dist = make_dist_mock(logits, torch.full((B,), -1.0))
        result = log_prob_reward(dist, torch.zeros(S, K), log_threshold=0.0)
        assert torch.all(result == -1)

    def test_4d_logits_expands_goal(self):
        logits = torch.randn(B, T, S, K)
        dist = make_dist_mock(logits, torch.ones(B, T))
        goal = torch.zeros(B, S, K)
        log_prob_reward(dist, goal, log_threshold=0.0)
        called_goal = dist.log_prob.call_args[0][0]
        assert called_goal.shape == (B, T, S, K)

    def test_4d_log_prob_above_threshold(self):
        logits = torch.randn(B, T, S, K)
        dist = make_dist_mock(logits, torch.full((B, T), 1.0))
        result = log_prob_reward(dist, torch.zeros(B, S, K), log_threshold=0.0)
        assert torch.all(result == 0)

    def test_4d_log_prob_below_threshold(self):
        logits = torch.randn(B, T, S, K)
        dist = make_dist_mock(logits, torch.full((B, T), -1.0))
        result = log_prob_reward(dist, torch.zeros(B, S, K), log_threshold=0.0)
        assert torch.all(result == -1)

    def test_output_shape_3d(self):
        logits = torch.randn(B, S, K)
        dist = make_dist_mock(logits, torch.ones(B))
        assert log_prob_reward(dist, torch.zeros(S, K), log_threshold=0.0).shape == (B, 1)

    def test_output_shape_4d(self):
        logits = torch.randn(B, T, S, K)
        dist = make_dist_mock(logits, torch.ones(B, T))
        assert log_prob_reward(dist, torch.zeros(B, S, K), log_threshold=0.0).shape == (B, T, 1)


class TestProbReward:
    def test_3d_returns_exp_of_log_prob(self):
        logits = torch.randn(B, S, K)
        log_prob_val = torch.full((B,), -1.0)
        dist = make_dist_mock(logits, log_prob_val)
        result = prob_reward(dist, torch.zeros(S, K))
        expected = log_prob_val.exp().unsqueeze(-1)
        assert torch.allclose(result, expected)

    def test_4d_returns_exp_of_log_prob(self):
        logits = torch.randn(B, T, S, K)
        log_prob_val = torch.full((B, T), -2.0)
        dist = make_dist_mock(logits, log_prob_val)
        result = prob_reward(dist, torch.zeros(B, S, K))
        expected = log_prob_val.exp().unsqueeze(-1)
        assert torch.allclose(result, expected)

    def test_4d_logits_expands_goal(self):
        logits = torch.randn(B, T, S, K)
        dist = make_dist_mock(logits, torch.ones(B, T))
        goal = torch.zeros(B, S, K)
        prob_reward(dist, goal)
        called_goal = dist.log_prob.call_args[0][0]
        assert called_goal.shape == (B, T, S, K)

    def test_output_shape_3d(self):
        logits = torch.randn(B, S, K)
        dist = make_dist_mock(logits, torch.ones(B))
        assert prob_reward(dist, torch.zeros(S, K)).shape == (B, 1)

    def test_output_shape_4d(self):
        logits = torch.randn(B, T, S, K)
        dist = make_dist_mock(logits, torch.ones(B, T))
        assert prob_reward(dist, torch.zeros(B, S, K)).shape == (B, T, 1)

    def test_range_zero_to_one(self):
        logits = torch.randn(B, S, K)
        log_prob_val = torch.tensor([-5.0, -0.1])
        dist = make_dist_mock(logits, log_prob_val)
        result = prob_reward(dist, torch.zeros(S, K))
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_log_prob_zero_gives_reward_one(self):
        logits = torch.randn(B, S, K)
        dist = make_dist_mock(logits, torch.zeros(B))
        result = prob_reward(dist, torch.zeros(S, K))
        assert torch.allclose(result, torch.ones(B, 1))


class TestArgmaxFullReward:
    def test_3d_argmax_matches_goal(self):
        logit = torch.zeros(B, S, K)
        logit[:, :, 0] = 1.0
        goal = F.one_hot(torch.zeros(S, dtype=torch.long), K).float()
        result = argmax_full_reward(logit, goal)
        assert torch.all(result == 0)

    def test_3d_argmax_mismatches_goal(self):
        logit = torch.zeros(B, S, K)
        logit[:, :, 0] = 1.0
        goal = F.one_hot(torch.ones(S, dtype=torch.long), K).float()
        result = argmax_full_reward(logit, goal)
        assert torch.all(result == -1)

    def test_3d_argmax_output_shape(self):
        logit = torch.zeros(B, S, K)
        goal = F.one_hot(torch.zeros(S, dtype=torch.long), K).float()
        assert argmax_full_reward(logit, goal).shape == (B, 1)

    def test_4d_argmax_matches_goal(self):
        logit = torch.zeros(B, T, S, K)
        logit[:, :, :, 0] = 1.0
        goal = F.one_hot(torch.zeros(B, S, dtype=torch.long), K).float()
        result = argmax_full_reward(logit, goal)
        assert torch.all(result == 0)

    def test_4d_argmax_mismatches_goal(self):
        logit = torch.zeros(B, T, S, K)
        logit[:, :, :, 0] = 1.0
        goal = F.one_hot(torch.ones(B, S, dtype=torch.long), K).float()
        result = argmax_full_reward(logit, goal)
        assert torch.all(result == -1)

    def test_4d_argmax_output_shape(self):
        logit = torch.zeros(B, T, S, K)
        goal = F.one_hot(torch.zeros(B, S, dtype=torch.long), K).float()
        assert argmax_full_reward(logit, goal).shape == (B, T, 1)


class TestMaxCosineReward:
    def test_3d_output_shape(self):
        state = torch.randn(B, S, K)
        goal = torch.randn(S, K)
        assert max_cosine_reward(state, goal).shape == (B, 1)

    def test_4d_output_shape(self):
        state = torch.randn(B, T, S, K)
        goal = torch.randn(B, S, K)
        assert max_cosine_reward(state, goal).shape == (B, T, 1)

    def test_3d_identical_gives_one(self):
        state = torch.randn(B, S, K)
        # goal per batch element equal to the state -> reward 1.0
        result = max_cosine_reward(state, state)
        assert torch.allclose(result, torch.ones(B, 1), atol=1e-5)

    def test_4d_identical_gives_one(self):
        goal = torch.randn(B, S, K)
        # state constant over time and equal to the goal -> reward 1.0
        state = goal.unsqueeze(1).expand(B, T, S, K).contiguous()
        result = max_cosine_reward(state, goal)
        assert torch.allclose(result, torch.ones(B, T, 1), atol=1e-5)

    def test_3d_orthogonal_gives_zero(self):
        # Disjoint one-hot cells -> dot product is 0.
        state = F.one_hot(torch.zeros(B, S, dtype=torch.long), K).float()
        goal = F.one_hot(torch.ones(S, dtype=torch.long), K).float()
        result = max_cosine_reward(state, goal)
        assert torch.allclose(result, torch.zeros(B, 1), atol=1e-6)

    def test_3d_matches_closed_form(self):
        state = torch.randn(B, S, K)
        goal = torch.randn(B, S, K)
        result = max_cosine_reward(state, goal)
        s = state.flatten(1)
        g = goal.flatten(1)
        dot = (s * g).sum(-1, keepdim=True)
        m = torch.maximum(s.norm(dim=-1, keepdim=True), g.norm(dim=-1, keepdim=True))
        expected = dot / m.pow(2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_shared_goal_3d(self):
        # goal (S, K) broadcasts across the batch like the other rewards.
        state = torch.randn(B, S, K)
        goal = torch.randn(S, K)
        result = max_cosine_reward(state, goal)
        s = state.flatten(1)
        g = goal.reshape(-1)
        dot = (s * g).sum(-1, keepdim=True)
        m = torch.maximum(s.norm(dim=-1, keepdim=True), g.norm())
        expected = dot / m.pow(2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_invalid_dim_raises(self):
        state = torch.zeros(B, K)
        goal = torch.zeros(K)
        with pytest.raises(ValueError):
            max_cosine_reward(state, goal)
