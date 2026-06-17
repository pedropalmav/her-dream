"""Tests for `Dreamer.imagine_goal` (goal_sample="imagination").

Covers the behavior introduced/parameterized on the goal-imagination branch:
- the rollout runs exactly `goal_imag_horizon` prior steps (the configurable knob),
- it uses uniformly random actions, not the actor,
- the returned goal has shape (N, S, K) and is a valid one-hot z,
- `goal_type="argmax_full"` returns the mode of the final prior instead of a sample.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.dreamer.conftest import K, N, S


def is_one_hot(z):
    """True iff `z` is one-hot over its last dim (exactly one 1, rest 0)."""
    return torch.all(z.sum(dim=-1) == 1) and torch.all((z == 0) | (z == 1))


class TestOutputShape:
    def test_full_returns_N_S_K(self, make_agent, obs):
        goal = make_agent(goal_type="full").imagine_goal(obs)
        assert goal.shape == (N, S, K)

    def test_argmax_full_returns_N_S_K(self, make_agent, obs):
        goal = make_agent(goal_type="argmax_full").imagine_goal(obs)
        assert goal.shape == (N, S, K)


class TestGoalIsOneHot:
    @pytest.mark.parametrize("goal_type", ["full", "argmax_full"])
    def test_goal_is_one_hot(self, make_agent, obs, goal_type):
        goal = make_agent(goal_type=goal_type).imagine_goal(obs)
        assert is_one_hot(goal)


class TestHorizonControlsRolloutLength:
    """The branch's feature: `goal_imag_horizon` sets the number of prior steps."""

    @pytest.mark.parametrize("horizon", [0, 1, 5, 15, 30])
    def test_img_step_called_horizon_times(self, make_agent, obs, rssm, horizon, monkeypatch):
        spy = _spy_on(monkeypatch, rssm, "img_step")
        make_agent(goal_imag_horizon=horizon).imagine_goal(obs)
        assert spy.count == horizon

    def test_horizon_zero_returns_posterior_z(self, make_agent, obs, rssm, monkeypatch):
        """With horizon 0 the goal is the posterior z (no prior step taken)."""
        spy = _spy_on(monkeypatch, rssm, "obs_step")
        goal = make_agent(goal_imag_horizon=0).imagine_goal(obs)
        assert spy.count == 1  # only the single posterior obs_step ran
        torch.testing.assert_close(goal, spy.last_return[0])

    def test_default_and_override_differ_in_call_count(self, make_agent, obs, rssm, monkeypatch):
        spy = _spy_on(monkeypatch, rssm, "img_step")
        make_agent(goal_imag_horizon=3).imagine_goal(obs)
        first = spy.count
        make_agent(goal_imag_horizon=10).imagine_goal(obs)
        assert first == 3 and spy.count - first == 10


class TestUsesRandomActions:
    def test_random_action_called_once_per_step(self, make_agent, obs):
        agent = make_agent(goal_imag_horizon=7)
        calls = {"n": 0}
        real = agent._random_action

        def counting(b):
            calls["n"] += 1
            return real(b)

        agent._random_action = counting
        agent.imagine_goal(obs)
        assert calls["n"] == 7

    def test_random_actions_are_valid_one_hot(self, make_agent, obs):
        # _random_action feeds the prior; assert it yields env-valid discrete actions.
        action = make_agent()._random_action(N)
        assert action.shape == (N, 4)
        assert is_one_hot(action)


class TestArgmaxFullMatchesFinalPriorMode:
    def test_goal_equals_argmax_of_final_logit(self, make_agent, obs, rssm, monkeypatch):
        spy = _spy_on(monkeypatch, rssm, "img_step")
        goal = make_agent(goal_type="argmax_full", goal_imag_horizon=4).imagine_goal(obs)
        final_logit = spy.last_return[2]
        expected = F.one_hot(torch.argmax(final_logit, dim=-1), num_classes=K).float()
        torch.testing.assert_close(goal, expected)


class TestNoGrad:
    @pytest.mark.parametrize("goal_type", ["full", "argmax_full"])
    def test_goal_does_not_require_grad(self, make_agent, obs, goal_type):
        # imagine_goal is @torch.no_grad: goals are detached targets, not learnable.
        goal = make_agent(goal_type=goal_type).imagine_goal(obs)
        assert not goal.requires_grad


# ---------------------------------------------------------------------------
# Helper: wrap a real RSSM method to record call count and last return value
# while keeping the genuine dynamics.
# ---------------------------------------------------------------------------


class _Spy:
    def __init__(self):
        self.count = 0
        self.last_return = None


def _spy_on(monkeypatch, obj, method_name):
    spy = _Spy()
    real = getattr(obj, method_name)

    def wrapper(*args, **kwargs):
        spy.count += 1
        spy.last_return = real(*args, **kwargs)
        return spy.last_return

    monkeypatch.setattr(obj, method_name, wrapper)
    return spy
