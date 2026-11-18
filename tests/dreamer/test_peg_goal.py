"""Tests for `Dreamer.peg_goal` (goal_sample="peg") on a real tiny LEXA agent."""

import pytest
import torch

import her_dream.peg as peg
from tests.dreamer.conftest import RK, RS, make_default_obs, make_real_dreamer

N = 2  # envs needing a goal


def lexa_dreamer(candidates=7, repeats=1, horizon=3, **overrides):
    agent, goal_shape = make_real_dreamer(
        lexa=True,
        model__imag_reward_source="temporal",
        model__peg__num_candidates=candidates,
        model__peg__repeats=repeats,
        model__peg__horizon=horizon,
        **overrides,
    )
    return agent, goal_shape


def first_obs(goal_shape):
    obs = make_default_obs(B=N, goal_shape=goal_shape)
    obs["is_first"] = torch.ones(N, dtype=torch.bool)
    return obs


def capture_candidates(monkeypatch):
    """Record the candidates `peg_goal` draws."""
    seen = {}
    real = peg.random_candidates

    def spy(*args, **kwargs):
        seen["cand"] = real(*args, **kwargs)
        return seen["cand"]

    monkeypatch.setattr(peg, "random_candidates", spy)
    return seen


class TestOutput:
    def test_returns_one_valid_one_hot_goal_per_env(self):
        agent, goal_shape = lexa_dreamer()
        goal, _ = agent.peg_goal(first_obs(goal_shape))
        assert goal.shape == (N, RS, RK)
        assert torch.all(goal.sum(-1) == 1)

    def test_first_row_goal_shape(self):
        agent, goal_shape = lexa_dreamer(goal_type="first_row")
        goal, _ = agent.peg_goal(first_obs(goal_shape))
        assert goal.shape == (N, RK)

    def test_metrics_are_ordered(self):
        agent, goal_shape = lexa_dreamer()
        _, metrics = agent.peg_goal(first_obs(goal_shape))
        assert set(metrics) == {"score_best", "score_mean", "score_std", "score_gap"}
        assert metrics["score_best"] >= metrics["score_mean"]
        assert metrics["score_gap"] >= 0


class TestSelection:
    def test_picks_the_highest_scoring_candidate_per_env(self, monkeypatch):
        C = 7
        agent, goal_shape = lexa_dreamer(candidates=C)
        seen = capture_candidates(monkeypatch)
        # Rollout batch is ordered (env, candidate): plant the best at index 4 for
        # env 0 and index 1 for env 1.
        planted = torch.zeros(N, C)
        planted[0, 4] = 1.0
        planted[1, 1] = 1.0
        agent._peg_score = lambda feat: planted.reshape(-1)
        goal, _ = agent.peg_goal(first_obs(goal_shape))
        cand = seen["cand"].reshape(N, C, RS, RK)
        torch.testing.assert_close(goal[0], cand[0, 4])
        torch.testing.assert_close(goal[1], cand[1, 1])

    def test_scores_are_averaged_over_repeats(self, monkeypatch):
        C, M = 3, 2
        agent, goal_shape = lexa_dreamer(candidates=C, repeats=M)
        seen = capture_candidates(monkeypatch)
        # (env, candidate, repeat): candidate 0 has the single highest sample but
        # candidate 2 the highest mean.
        scores = torch.tensor([[[10.0, -10.0], [0.0, 0.0], [3.0, 3.0]]] * N)
        agent._peg_score = lambda feat: scores.reshape(-1)
        goal, _ = agent.peg_goal(first_obs(goal_shape))
        cand = seen["cand"].reshape(N, C, RS, RK)
        torch.testing.assert_close(goal[0], cand[0, 2])

    def test_default_score_is_the_explorer_value(self):
        agent, goal_shape = lexa_dreamer()
        feat = torch.randn(5, agent.rssm.feat_size)
        expected = agent.explorer._frozen_value(feat).mode.squeeze(-1)
        torch.testing.assert_close(agent._peg_score(feat), expected)


class TestRollout:
    @pytest.mark.parametrize("horizon", [0, 1, 4])
    def test_runs_horizon_prior_steps_on_every_candidate(self, monkeypatch, horizon):
        C, M = 5, 2
        agent, goal_shape = lexa_dreamer(candidates=C, repeats=M, horizon=horizon)
        batches = []
        real = agent._frozen_rssm.img_step

        def spy(stoch, deter, action):
            batches.append(stoch.shape[0])
            return real(stoch, deter, action)

        monkeypatch.setattr(agent._frozen_rssm, "img_step", spy)
        agent.peg_goal(first_obs(goal_shape))
        assert batches == [N * C * M] * horizon

    def test_achiever_is_conditioned_on_the_candidates(self, monkeypatch):
        agent, goal_shape = lexa_dreamer(candidates=4, horizon=2)
        seen = capture_candidates(monkeypatch)
        goals_seen = []
        real = agent.ac.frozen_policy

        def spy(feat, goal):
            goals_seen.append(goal)
            return real(feat, goal)

        monkeypatch.setattr(agent.ac, "frozen_policy", spy)
        agent.peg_goal(first_obs(goal_shape))
        assert len(goals_seen) == 2
        torch.testing.assert_close(goals_seen[0], seen["cand"])

    def test_leaves_parameters_untouched(self):
        agent, goal_shape = lexa_dreamer()
        before = {k: v.clone() for k, v in agent.state_dict().items()}
        agent.peg_goal(first_obs(goal_shape))
        for k, v in agent.state_dict().items():
            torch.testing.assert_close(v, before[k])


class TestHorizonConfig:
    def test_null_horizon_follows_the_go_phase(self):
        agent, _ = make_real_dreamer(lexa=True, env__time_limit=40, model__peg__go_ratio=0.25)
        assert agent.peg_horizon == 10


def test_requires_lexa():
    agent, goal_shape = make_real_dreamer()
    with pytest.raises(ValueError, match="lexa=True"):
        agent.peg_goal(first_obs(goal_shape))


class TestGivenCandidates:
    def test_chooses_among_the_given_candidates(self, monkeypatch):
        C = 3
        agent, goal_shape = lexa_dreamer(candidates=99)
        cand = torch.nn.functional.one_hot(torch.randint(0, RK, (N, C, RS)), RK).float()
        drawn = capture_candidates(monkeypatch)
        planted = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        agent._peg_score = lambda feat: planted.reshape(-1)
        goal, _ = agent.peg_goal(first_obs(goal_shape), candidates=cand)
        assert "cand" not in drawn  # no random draw when candidates are given
        torch.testing.assert_close(goal[0], cand[0, 2])
        torch.testing.assert_close(goal[1], cand[1, 0])
