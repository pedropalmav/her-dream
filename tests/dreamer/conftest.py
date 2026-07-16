"""Fixtures for exercising the goal-sampling slices of `Dreamer`.

Both `Dreamer.imagine_goal` (goal_sample="imagination") and
`Dreamer.encode_observation` (goal_sample="image") only touch a small slice of
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
from tensordict import TensorDict

import goals
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

    The real `imagine_goal`, `encode_observation` and `_random_action` are
    borrowed verbatim so the tests run the actual implementation, not a
    reimplementation.
    """

    # Real implementations under test.
    imagine_goal = Dreamer.imagine_goal
    encode_observation = Dreamer.encode_observation
    _random_action = Dreamer._random_action

    def __init__(self, rssm, *, goal_imag_horizon, goal_type, act_dim=A, embed_size=E, device="cpu"):
        self._frozen_rssm = rssm
        self.goal_imag_horizon = goal_imag_horizon
        self.goal_type = goal_type
        self._goal_spec = goals.make_goal_spec(
            SimpleNamespace(goal_type=goal_type, **goals.default_descriptors(goal_type))
        )
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
    """A rendered-goal obs as encode_observation expects: image + one-hot direction."""
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


import pathlib  # noqa: E402

from hydra import compose, initialize_config_dir  # noqa: E402

from rewards import make_reward  # noqa: E402

# Small real-build dims. IMG must survive the 4-stage CNN downsample (minres=4).
IMG = 64
RS, RK = 8, 8  # rssm stoch groups / categories
RDETER = 32  # rssm deter dim (divisible by blocks=4)
ACT = 5  # discrete action dim
RB, RT = 2, 4  # batch, time for a sampled chunk

_CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[2] / "configs")

# Overrides shrinking every size knob so a real update() is fast on CPU.
_SMALL_OVERRIDES = [
    "device=cpu",
    f"model.rssm.stoch={RS}",
    f"model.rssm.discrete={RK}",
    f"model.deter={RDETER}",
    "model.rssm.blocks=4",
    "model.hidden=16",
    "model.units=16",
    "model.depth=2",
    "model.rssm.img_layers=1",
    "model.rssm.obs_layers=1",
    "model.rssm.dyn_layers=1",
    "model.imag_horizon=2",
    "model.horizon=5",
]


def build_model_config(**dotlist_overrides):
    """Compose the real Hydra tree and return the (fresh) `model` node.

    A fresh config must be composed per `Dreamer` because `Dreamer.__init__`
    mutates `config.actor.shape`/`config.actor.dist` in place.

    ``dotlist_overrides`` are passed as ``key=value`` and may use dotted keys via
    double-underscore, e.g. ``model__rep_loss="dreamer"`` -> ``model.rep_loss=dreamer``.
    """
    extra = [f"{k.replace('__', '.')}={v}" for k, v in dotlist_overrides.items()]
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name="configs", overrides=[*_SMALL_OVERRIDES, *extra])


class _Space:
    """Minimal gym-space stand-in: only `.shape` is read by `Dreamer`."""

    def __init__(self, shape):
        self.shape = tuple(shape)


class _ObsSpace:
    def __init__(self, spaces):
        self.spaces = spaces


def make_obs_space(goal_shape=(RS, RK)):
    """Obs space with the keys the random_goal encoder consumes plus `goal`."""
    return _ObsSpace({
        "image": _Space((IMG, IMG, 3)),
        "direction": _Space((4,)),
        "goal": _Space(goal_shape),
    })


def act_discrete(n=ACT):
    """One-hot discrete action space: `.shape=(n,)`, `.discrete=True`, no `.n`."""
    return SimpleNamespace(shape=(n,), discrete=True)


def act_multi(nvec=(2, 3)):
    """Multi-discrete action space: `.shape=nvec`, `.multi_discrete=True`."""
    return SimpleNamespace(shape=tuple(nvec), multi_discrete=True)


def act_cont(n=ACT):
    """Continuous action space: plain `.shape`, no discrete/multi attrs."""
    return SimpleNamespace(shape=(n,))


def act_with_n(n=ACT):
    """Gym-Discrete-like space exposing `.n` (and an empty `.shape`)."""
    return SimpleNamespace(n=n, shape=())


class StubReplayBuffer:
    """Yields a single fixed-shape batch and records the `update` write-back.

    Mirrors the real buffer contract used by `Dreamer.update`:
    `sample() -> (TensorDict, index, initial)` and
    `update(index, stoch, deter)`.
    """

    def __init__(self, goal_shape=(RS, RK), mission=False, act_dim=ACT, B=RB, T=RT):
        self.goal_shape = tuple(goal_shape)
        self.mission = mission
        self.act_dim = act_dim
        self.B = B
        self.T = T
        self.update_calls = []

    def sample(self):
        B, T = self.B, self.T
        action = torch.zeros(B, T, self.act_dim)
        action[..., 0] = 1.0
        goal = torch.zeros(B, T, *self.goal_shape)
        goal[..., 0] = 1.0  # valid one-hot along the last (category) dim
        data = {
            "image": torch.randint(0, 256, (B, T, IMG, IMG, 3), dtype=torch.uint8),
            "direction": torch.zeros(B, T, 4),
            "action": action,
            "goal": goal,
            "is_first": torch.zeros(B, T, dtype=torch.bool),
            "is_last": torch.zeros(B, T, 1),
            "is_terminal": torch.zeros(B, T, 1),
            "reward": torch.zeros(B, T, 1),
        }
        if self.mission:
            data["mission"] = torch.zeros(B, T, 20, dtype=torch.int8)
        td = TensorDict(data, batch_size=(B, T))
        index = torch.arange(B)
        initial = (torch.zeros(B, RS, RK), torch.zeros(B, RDETER))
        return td, index, initial

    def update(self, index, stoch, deter):
        self.update_calls.append((index, stoch, deter))


def make_real_dreamer(goal_type="full", act="discrete", **cfg_overrides):
    """Build a genuine tiny `Dreamer` and return `(agent, goal_shape)`.

    `act` selects the action-space kind: "discrete" | "multi" | "cont" | "n".
    `cfg_overrides` are forwarded to `build_model_config` (double-underscore ->
    dotted key), e.g. ``model__rep_loss="dreamer"``.
    """
    cfg = build_model_config(goal_type=goal_type, **cfg_overrides)
    act_space = {
        "discrete": act_discrete,
        "multi": act_multi,
        "cont": act_cont,
        "n": act_with_n,
    }[act]()
    goal_shape = (RK,) if goal_type == "first_row" else (RS, RK)
    obs_space = make_obs_space(goal_shape)
    agent = Dreamer(cfg.model, obs_space, act_space, reward_function=make_reward(cfg.model))
    return agent, goal_shape


def make_default_obs(B=2, goal_shape=(RS, RK)):
    """A single-step obs for `act`: image + one-hot direction + one-hot goal."""
    goal = torch.zeros(B, *goal_shape)
    goal[..., 0] = 1.0
    return {
        "image": torch.randint(0, 256, (B, IMG, IMG, 3), dtype=torch.uint8),
        "direction": torch.zeros(B, 4),
        "goal": goal,
        "is_first": torch.zeros(B, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Fixtures for the common "default" agent.
#
# Most tests need a varied build and so call `make_real_dreamer(...)` with
# overrides directly (that is what the factory is for). These fixtures cover the
# frequent no-argument case — the discrete, goal_type="full" agent — so those
# tests read as plain pytest. They are function-scoped: `Dreamer` is stateful
# (update() mutates params/optimizer/EMA, and some tests mutate params), so a
# shared instance across tests would leak state.
# ---------------------------------------------------------------------------


@pytest.fixture
def default_dreamer():
    """The default discrete, goal_type="full" `Dreamer` (goal_shape dropped)."""
    agent, _ = make_real_dreamer()
    return agent


@pytest.fixture
def default_obs():
    """A single-step obs matching the default agent, for `act`."""
    return make_default_obs()


@pytest.fixture
def default_buffer():
    """A `StubReplayBuffer` matching the default agent (discrete, full goal)."""
    return StubReplayBuffer((RS, RK), act_dim=ACT)
