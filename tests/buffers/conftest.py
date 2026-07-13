from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

import her_dream.goals as goals

S, K, D, A = 8, 8, 16, 3


def make_buffer_config(**overrides):
    defaults = dict(device="cpu", storage_device="cpu", batch_size=2, batch_length=4, max_size=40)
    return SimpleNamespace(**{**defaults, **overrides})


def make_her_config(**overrides):
    defaults = dict(
        device="cpu",
        storage_device="cpu",
        batch_size=2,
        batch_length=4,
        max_size=40,
        goal_type="full",
        her_ratio=0.5,
        her_strategy="future",
        n_envs=2,
    )
    merged = {**defaults, **overrides}
    for key, val in goals.default_descriptors(merged["goal_type"]).items():
        merged.setdefault(key, val)
    return SimpleNamespace(**merged)


def make_transition(n_envs=2):
    return TensorDict(
        {
            "stoch": torch.zeros(n_envs, S, K),
            "deter": torch.zeros(n_envs, D),
            "action": torch.zeros(n_envs, A),
            "reward": torch.zeros(n_envs),
            "is_last": torch.zeros(n_envs, dtype=torch.bool),
            "goal": torch.zeros(n_envs, K),
            "logit": torch.zeros(n_envs, S, K),
            "env": torch.arange(n_envs),
        },
        batch_size=[n_envs],
    )


def make_her_buffer(**config_overrides):
    from her_dream.buffers.her_buffer import HERBuffer

    config = make_her_config(**config_overrides)

    def dummy_reward(achieved, desired):
        return torch.zeros(achieved.shape[0])

    return HERBuffer(config, dummy_reward)


def make_mock_buffer_for_goal(n_steps):
    stoch = torch.ones(n_steps, S, K)
    logit = torch.ones(n_steps, S, K)
    td = TensorDict({"stoch": stoch, "logit": logit}, batch_size=[n_steps])
    mock = MagicMock()
    mock.__getitem__ = MagicMock(return_value=td)
    return mock


@pytest.fixture
def buf_config():
    return make_buffer_config()


@pytest.fixture
def buffer(buf_config):
    from her_dream.buffers.buffer import Buffer

    return Buffer(buf_config)


@pytest.fixture
def filled_buffer(buffer):
    for _ in range(6):
        buffer.add_transition(make_transition(n_envs=2))
    return buffer


@pytest.fixture
def her_buffer():
    return make_her_buffer()


@pytest.fixture
def filled_her_buffer():
    buf = make_her_buffer()
    for _ in range(6):
        buf.add_transition(make_transition())
    return buf
