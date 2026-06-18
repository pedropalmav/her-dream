"""Tests for `Dreamer.encode_observation` (goal_sample="image").

`encode_observation` encodes a rendered goal observation into a goal z with a SINGLE
posterior step from the initial latent state — no prior rollout, unlike
`imagine_goal`. Covers:
- the returned goal has shape (N, S, K) and is a valid one-hot z,
- exactly one posterior step runs and no prior (`img_step`) step is taken,
- `goal_type="full"` returns the posterior sample, `"argmax_full"` its mode,
- the batch dim follows the obs, and goals are detached (no grad).
"""

import pytest
import torch
import torch.nn.functional as F

from tests.dreamer.conftest import K, N, S, make_goal_image_obs


def is_one_hot(z):
    """True iff `z` is one-hot over its last dim (exactly one 1, rest 0)."""
    return torch.all(z.sum(dim=-1) == 1) and torch.all((z == 0) | (z == 1))


class TestOutputShape:
    def test_full_returns_N_S_K(self, make_agent, goal_image_obs):
        goal = make_agent(goal_type="full").encode_observation(goal_image_obs)
        assert goal.shape == (N, S, K)

    def test_argmax_full_returns_N_S_K(self, make_agent, goal_image_obs):
        goal = make_agent(goal_type="argmax_full").encode_observation(goal_image_obs)
        assert goal.shape == (N, S, K)

    @pytest.mark.parametrize("n", [1, 5, 8])
    def test_batch_dim_follows_obs(self, make_agent, n):
        goal = make_agent(goal_type="full").encode_observation(make_goal_image_obs(n))
        assert goal.shape == (n, S, K)


class TestGoalIsOneHot:
    @pytest.mark.parametrize("goal_type", ["full", "argmax_full"])
    def test_goal_is_one_hot(self, make_agent, goal_image_obs, goal_type):
        goal = make_agent(goal_type=goal_type).encode_observation(goal_image_obs)
        assert is_one_hot(goal)


class TestSinglePosteriorStep:
    """A single posterior step maps the image to z — no imagination rollout."""

    def test_obs_step_called_once(self, make_agent, goal_image_obs, rssm, monkeypatch):
        spy = _spy_on(monkeypatch, rssm, "obs_step")
        make_agent(goal_type="full").encode_observation(goal_image_obs)
        assert spy.count == 1

    def test_no_prior_step_taken(self, make_agent, goal_image_obs, rssm, monkeypatch):
        spy = _spy_on(monkeypatch, rssm, "img_step")
        make_agent(goal_type="full").encode_observation(goal_image_obs)
        assert spy.count == 0


class TestFullReturnsPosteriorSample:
    def test_goal_equals_posterior_stoch(self, make_agent, goal_image_obs, rssm, monkeypatch):
        spy = _spy_on(monkeypatch, rssm, "obs_step")
        goal = make_agent(goal_type="full").encode_observation(goal_image_obs)
        # obs_step returns (stoch, deter, logit); full uses the sampled stoch.
        torch.testing.assert_close(goal, spy.last_return[0])


class TestArgmaxFullReturnsPosteriorMode:
    def test_goal_equals_argmax_of_posterior_logit(self, make_agent, goal_image_obs, rssm, monkeypatch):
        spy = _spy_on(monkeypatch, rssm, "obs_step")
        goal = make_agent(goal_type="argmax_full").encode_observation(goal_image_obs)
        posterior_logit = spy.last_return[2]
        expected = F.one_hot(torch.argmax(posterior_logit, dim=-1), num_classes=K).float()
        torch.testing.assert_close(goal, expected)


class TestNoGrad:
    @pytest.mark.parametrize("goal_type", ["full", "argmax_full"])
    def test_goal_does_not_require_grad(self, make_agent, goal_image_obs, goal_type):
        # encode_observation is @torch.no_grad: goals are detached targets, not learnable.
        goal = make_agent(goal_type=goal_type).encode_observation(goal_image_obs)
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
