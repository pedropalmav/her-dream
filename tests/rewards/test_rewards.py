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
    row_by_row_reward,
)

from .conftest import B, K, S, T, make_dist_mock


class TestMakeReward:
    def test_first_row_type(self):
        config = SimpleNamespace(goal_type="first_row")
        assert make_reward(config) is first_row_reward

    def test_row_by_row_type(self):
        config = SimpleNamespace(goal_type="row_by_row")
        assert make_reward(config) is row_by_row_reward

    def test_full_type(self):
        config = SimpleNamespace(goal_type="full")
        assert make_reward(config) is full_goal_reward

    def test_argmax_full_type(self):
        config = SimpleNamespace(goal_type="argmax_full")
        assert make_reward(config) is argmax_full_reward

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

    def test_prob_type(self):
        config = SimpleNamespace(goal_type="prob")
        assert make_reward(config) is prob_reward

    def test_max_cosine_type(self):
        config = SimpleNamespace(goal_type="max_cosine")
        assert make_reward(config) is max_cosine_reward

    def test_invalid_type_raises(self):
        config = SimpleNamespace(goal_type="unknown")
        with pytest.raises(ValueError):
            make_reward(config)


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
