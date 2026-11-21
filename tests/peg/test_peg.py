"""Tests for the network-free helpers of PEG goal selection (`her_dream/peg.py`)."""

from types import SimpleNamespace

import pytest
import torch

import her_dream.goals as goals
from her_dream.peg import explore_every_ep, go_steps, planning_horizon, random_candidates, refit

S, K = 6, 5


def spec(goal_type):
    return goals.make_goal_spec(SimpleNamespace(goal_type=goal_type, **goals.default_descriptors(goal_type)))


class TestGoSteps:
    @pytest.mark.parametrize(
        ("ratio", "time_limit", "repeat", "expected"),
        [(0.5, 100, 1, 50), (0.3, 500, 1, 150), (0.5, 100, 2, 25), (0.0, 100, 1, 0), (1.0, 100, 1, 100)],
    )
    def test_is_a_fraction_of_the_agent_steps(self, ratio, time_limit, repeat, expected):
        assert go_steps(ratio, time_limit, repeat) == expected

    @pytest.mark.parametrize("ratio", [-0.1, 1.5])
    def test_rejects_ratios_outside_the_unit_interval(self, ratio):
        with pytest.raises(ValueError, match="go_ratio"):
            go_steps(ratio, 100, 1)


class TestPlanningHorizon:
    @staticmethod
    def cfg(horizon):
        return SimpleNamespace(horizon=horizon, go_ratio=0.5, time_limit=100, action_repeat=1)

    def test_null_follows_the_go_phase(self):
        assert planning_horizon(self.cfg(None)) == 50

    def test_explicit_value_wins(self):
        assert planning_horizon(self.cfg(25)) == 25


class TestRandomCandidates:
    @pytest.mark.parametrize("goal_type", ["full", "argmax_full", "row_by_row"])
    def test_one_hot_types_get_one_category_per_group(self, goal_type):
        cand = random_candidates(spec(goal_type), 64, S, K, "cpu")
        assert cand.shape == (64, S, K)
        assert torch.all(cand.sum(-1) == 1) and torch.all((cand == 0) | (cand == 1))

    def test_first_row_is_a_single_group(self):
        cand = random_candidates(spec("first_row"), 64, S, K, "cpu")
        assert cand.shape == (64, K)
        assert torch.all(cand.sum(-1) == 1)

    def test_logit_goals_are_real_valued(self):
        cand = random_candidates(spec("max_cosine"), 64, S, K, "cpu")
        assert cand.shape == (64, S, K)
        assert not torch.all((cand == 0) | (cand == 1))

    def test_candidates_differ(self):
        cand = random_candidates(spec("full"), 64, S, K, "cpu")
        assert torch.unique(cand.reshape(64, -1), dim=0).shape[0] > 1


class TestExploreEveryEp:
    def test_lexa_keeps_its_split_by_default(self):
        assert explore_every_ep(None, "buffer") == 2

    def test_peg_has_no_explorer_only_episodes_by_default(self):
        assert explore_every_ep(None, "peg") == 0

    @pytest.mark.parametrize("goal_sample", ["peg", "buffer"])
    def test_explicit_value_wins(self, goal_sample):
        assert explore_every_ep(3, goal_sample) == 3


class TestRefit:
    N, C = 2, 5

    def samples(self):
        return torch.randn(self.N, self.C, 3, 4)

    def test_mppi_with_large_gamma_collapses_onto_the_best(self):
        x = self.samples()
        scores = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 2.0]])
        mean, std = refit("mppi", x, scores, mppi_gamma=1e4, elite_ratio=0.2)
        torch.testing.assert_close(mean[0], x[0, 1])
        torch.testing.assert_close(mean[1], x[1, 4])
        assert std.abs().max() < 1e-3

    def test_mppi_with_zero_gamma_is_the_plain_moments(self):
        x = self.samples()
        mean, std = refit("mppi", x, torch.randn(self.N, self.C), mppi_gamma=0.0, elite_ratio=0.2)
        torch.testing.assert_close(mean, x.mean(dim=1))
        torch.testing.assert_close(std, x.std(dim=1, unbiased=False))

    def test_mppi_matches_the_weighted_formula(self):
        x = self.samples()
        scores = torch.randn(self.N, self.C)
        w = torch.softmax(10.0 * scores, dim=-1)[..., None, None]
        mean, std = refit("mppi", x, scores, mppi_gamma=10.0, elite_ratio=0.2)
        torch.testing.assert_close(mean, (w * x).sum(1))
        torch.testing.assert_close(std, (w * (x - mean[:, None]) ** 2).sum(1).sqrt())

    def test_cem_refits_on_the_elites(self):
        x = self.samples()
        scores = torch.tensor([[5.0, 4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 9.0, 8.0]])
        mean, std = refit("cem", x, scores, mppi_gamma=10.0, elite_ratio=0.4)  # 2 elites
        torch.testing.assert_close(mean[0], x[0, :2].mean(0))
        torch.testing.assert_close(mean[1], x[1, 3:].mean(0))
        torch.testing.assert_close(std[0], (x[0, :2].var(0, unbiased=False) + 1e-6).sqrt())

    def test_cem_keeps_at_least_one_elite(self):
        x = self.samples()
        scores = torch.tensor([[0.0, 0.0, 3.0, 0.0, 0.0]] * self.N)
        mean, _ = refit("cem", x, scores, mppi_gamma=10.0, elite_ratio=0.01)
        torch.testing.assert_close(mean[0], x[0, 2])

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="refit"):
            refit("random", self.samples(), torch.zeros(self.N, self.C), 10.0, 0.2)
