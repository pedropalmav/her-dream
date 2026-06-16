"""Fixtures for exercising the goal-sampling slices of `Dreamer`.

Both `Dreamer.imagine_goal` (goal_sample="imagination") and
`Dreamer.observe_goal` (goal_sample="image") only touch a small slice of
`Dreamer`: the frozen RSSM, the frozen encoder, `preprocess`, the
`goal_imag_horizon` / `goal_type` attributes (and `_random_action` for the
imagination rollout). Building a full `Dreamer` (encoders, decoder, optimizer,
EMA copies, ...) just to test the goal logic would be slow and brittle, so we
borrow the real methods onto a minimal stub backed by a real `RSSM`. This keeps
the latent dynamics genuine while isolating the behavior under test.
"""

from types import SimpleNamespace

import pytest
import torch

from dreamer import Dreamer
from rssm import RSSM

N = 3  # number of envs needing a fresh goal
S, K = 8, 8  # stoch groups, categories per group
D = 32  # deter dim (must be divisible by blocks=4)
A = 4  # action dim (discrete -> one-hot of size A)
E = 16  # embed size
DEFAULT_HORIZON = 5


def make_rssm_config(**overrides):
    defaults = dict(
        stoch=S,
        discrete=K,
        deter=D,
        hidden=16,
        obs_layers=1,
        img_layers=1,
        dyn_layers=1,
        blocks=4,
        act="SiLU",
        unimix_ratio=0.01,
        initial="learned",
        device="cpu",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class StubDreamer:
    """Minimal object exposing only what the goal methods touch.

    The real `imagine_goal`, `observe_goal` and `_random_action` are borrowed
    verbatim so the tests run the actual implementation, not a reimplementation.
    """

    # Real implementations under test.
    imagine_goal = Dreamer.imagine_goal
    observe_goal = Dreamer.observe_goal
    _random_action = Dreamer._random_action

    def __init__(self, rssm, *, goal_imag_horizon, goal_type, act_dim=A, embed_size=E, device="cpu"):
        self._frozen_rssm = rssm
        self.goal_imag_horizon = goal_imag_horizon
        self.goal_type = goal_type
        self.act_dim = act_dim
        self.device = torch.device(device)
        self._embed_size = embed_size
        # Discrete action space of size `act_dim` -> _random_action yields one-hot.
        self._act_n = act_dim
        self._act_multi = False
        self._act_nvec = None

    def preprocess(self, obs):
        # The real preprocess only normalizes "image"; our encoder is a stub.
        return obs

    def _frozen_encoder(self, p_obs):
        # Stand-in for MultiEncoder: N is the batch dim of any obs entry, so this
        # works for both is_first (imagination) and image (image) goal obs.
        some = next(iter(p_obs.values()))
        n = some.shape[0]
        return torch.randn(n, self._embed_size, device=self.device)


@pytest.fixture
def rssm():
    return RSSM(make_rssm_config(), embed_size=E, act_dim=A).eval()


@pytest.fixture
def obs():
    # imagine_goal reads obs["is_first"] (shape[0] -> N) and passes it as the
    # posterior reset mask. True == start of episode, matching real usage.
    return {"is_first": torch.ones(N, dtype=torch.bool)}


def make_goal_image_obs(n=N):
    """A rendered-goal obs as observe_goal expects: image + one-hot direction."""
    direction = torch.zeros(n, 4, dtype=torch.float32)
    direction[:, 0] = 1.0
    return {
        "image": torch.zeros(n, 8, 8, 3, dtype=torch.uint8),
        "direction": direction,
    }


@pytest.fixture
def goal_image_obs():
    return make_goal_image_obs()


@pytest.fixture
def make_agent(rssm):
    """Factory: build a StubDreamer with the given horizon / goal_type."""

    def _make(goal_imag_horizon=DEFAULT_HORIZON, goal_type="full"):
        return StubDreamer(rssm, goal_imag_horizon=goal_imag_horizon, goal_type=goal_type)

    return _make
