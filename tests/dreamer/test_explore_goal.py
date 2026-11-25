"""Tests for `Dreamer.explore_goal` (goal_sample="explore_imagination[_refresh]").

Runs on a real tiny Plan2Explore / LEXA agent: the explorer is rolled out
`explore_goal.num_rollouts` times for `goal_imag_horizon` steps and the final z
of the rollout its critic values most becomes the goal.
"""

import pytest
import torch

from tests.dreamer.conftest import RK, RS, make_default_obs, make_real_dreamer

N = 2  # envs needing a goal


def explorer_dreamer(mode="lexa", rollouts=5, horizon=3, **overrides):
    extra = {"model__imag_reward_source": "temporal"} if mode == "lexa" else {}
    return make_real_dreamer(
        **{mode: True},
        model__goal_imag_horizon=horizon,
        model__explore_goal__num_rollouts=rollouts,
        **extra,
        **overrides,
    )


def first_obs(goal_shape):
    obs = make_default_obs(B=N, goal_shape=goal_shape)
    obs["is_first"] = torch.ones(N, dtype=torch.bool)
    return obs


def record_img_steps(monkeypatch, agent):
    """Record every img_step's batch size and output."""
    calls = []
    real = agent._frozen_rssm.img_step

    def spy(stoch, deter, action):
        out = real(stoch, deter, action)
        calls.append(out)
        return out

    monkeypatch.setattr(agent._frozen_rssm, "img_step", spy)
    return calls


class TestOutput:
    @pytest.mark.parametrize("mode", ["lexa", "plan2explore"])
    def test_returns_one_valid_one_hot_goal_per_env(self, mode):
        agent, goal_shape = explorer_dreamer(mode)
        goal, _ = agent.explore_goal(first_obs(goal_shape))
        assert goal.shape == (N, RS, RK)
        assert torch.all(goal.sum(-1) == 1)
        assert not goal.requires_grad

    def test_metrics_are_ordered(self):
        agent, goal_shape = explorer_dreamer()
        _, metrics = agent.explore_goal(first_obs(goal_shape))
        assert set(metrics) == {"score_best", "score_mean", "score_std", "score_gap"}
        assert metrics["score_best"] >= metrics["score_mean"]
        assert metrics["score_gap"] >= 0

    def test_single_rollout_has_zero_spread(self):
        agent, goal_shape = explorer_dreamer(rollouts=1)
        _, metrics = agent.explore_goal(first_obs(goal_shape))
        assert metrics["score_std"] == 0 and metrics["score_gap"] == 0


class TestSelection:
    def test_goal_is_the_final_z_of_the_best_rollout(self, monkeypatch):
        M = 5
        agent, goal_shape = explorer_dreamer(rollouts=M, horizon=2)
        calls = record_img_steps(monkeypatch, agent)
        # Batch ordered (env, rollout): best is rollout 3 for env 0, 0 for env 1.
        planted = torch.zeros(N, M)
        planted[0, 3] = 1.0
        planted[1, 0] = 1.0
        agent._peg_score = lambda feat: planted.reshape(-1)
        goal, _ = agent.explore_goal(first_obs(goal_shape))
        final_stoch = calls[-1][0].reshape(N, M, RS, RK)
        torch.testing.assert_close(goal[0], final_stoch[0, 3])
        torch.testing.assert_close(goal[1], final_stoch[1, 0])

    def test_argmax_goal_type_takes_the_mode_of_the_chosen_logit(self, monkeypatch):
        M = 4
        agent, goal_shape = explorer_dreamer(rollouts=M, horizon=2, goal_type="argmax_full")
        calls = record_img_steps(monkeypatch, agent)
        planted = torch.zeros(N, M)
        planted[:, 2] = 1.0
        agent._peg_score = lambda feat: planted.reshape(-1)
        goal, _ = agent.explore_goal(first_obs(goal_shape))
        logit = calls[-1][2].reshape(N, M, RS, RK)[:, 2]
        expected = torch.nn.functional.one_hot(logit.argmax(-1), RK).float()
        torch.testing.assert_close(goal, expected)

    def test_scores_the_final_states(self, monkeypatch):
        M = 3
        agent, goal_shape = explorer_dreamer(rollouts=M, horizon=2)
        calls = record_img_steps(monkeypatch, agent)
        scored = []
        real = agent._peg_score

        def spy(feat):
            scored.append(feat)
            return real(feat)

        agent._peg_score = spy
        agent.explore_goal(first_obs(goal_shape))
        stoch, deter, _ = calls[-1]
        torch.testing.assert_close(scored[0], agent._frozen_rssm.get_feat(stoch, deter))


class TestRollout:
    @pytest.mark.parametrize("horizon", [0, 1, 4])
    def test_runs_horizon_prior_steps_on_every_rollout(self, monkeypatch, horizon):
        M = 3
        agent, goal_shape = explorer_dreamer(rollouts=M, horizon=horizon)
        calls = record_img_steps(monkeypatch, agent)
        agent.explore_goal(first_obs(goal_shape))
        assert [c[0].shape[0] for c in calls] == [N * M] * horizon

    def test_uses_the_explorer_policy_not_the_achiever(self, monkeypatch):
        agent, goal_shape = explorer_dreamer(horizon=2)
        seen = []
        real = agent.explorer.frozen_policy

        def spy(feat, goal):
            seen.append(goal)
            return real(feat, goal)

        monkeypatch.setattr(agent.explorer, "frozen_policy", spy)
        monkeypatch.setattr(agent.ac, "frozen_policy", lambda *a: pytest.fail("achiever was rolled out"))
        agent.explore_goal(first_obs(goal_shape))
        assert seen == [None, None]

    def test_starts_from_the_given_state(self, monkeypatch):
        agent, goal_shape = explorer_dreamer(horizon=0)
        seen = {}
        real = agent._frozen_rssm.obs_step

        def spy(stoch, deter, prev_action, embed, reset):
            seen.update(deter=deter)
            return real(stoch, deter, prev_action, embed, reset)

        monkeypatch.setattr(agent._frozen_rssm, "obs_step", spy)
        state = agent.get_initial_state(N)
        state["deter"] = torch.ones_like(state["deter"])
        obs = first_obs(goal_shape)
        obs["is_first"] = torch.zeros(N, dtype=torch.bool)
        agent.explore_goal(obs, state=state)
        assert seen["deter"] is state["deter"]

    def test_leaves_parameters_untouched(self):
        agent, goal_shape = explorer_dreamer()
        before = {k: v.clone() for k, v in agent.state_dict().items()}
        agent.explore_goal(first_obs(goal_shape))
        for k, v in agent.state_dict().items():
            torch.testing.assert_close(v, before[k])


def test_requires_an_explorer():
    agent, goal_shape = make_real_dreamer()
    with pytest.raises(ValueError, match="plan2explore=True or lexa=True"):
        agent.explore_goal(first_obs(goal_shape))


def test_rejects_zero_rollouts():
    with pytest.raises(ValueError, match="num_rollouts"):
        explorer_dreamer(rollouts=0)
