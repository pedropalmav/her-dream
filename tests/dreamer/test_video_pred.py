"""Tests for `Dreamer.video_pred` (decoder-only rollout visualization).

`video_pred` reconstructs the first 5 steps with the posterior, then open-loop
imagines the rest with the prior, and stacks truth/model/error for display. It
requires a decoder, so it only works with `rep_loss="dreamer"` and needs a
sequence of at least 6 steps (it slices `[:, 5:]`).
"""

import pytest

from tests.dreamer.conftest import StubReplayBuffer, make_real_dreamer

T = 7  # must be > 5 for the open-loop slice to be non-empty


def _sample(agent, goal_shape):
    buf = StubReplayBuffer(goal_shape, act_dim=agent.act_dim, T=T)
    data, _, initial = buf.sample()
    return data, initial


class TestVideoPred:
    def test_output_shape(self):
        agent, gs = make_real_dreamer(model__rep_loss="dreamer")
        data, initial = _sample(agent, gs)
        out = agent.video_pred(dict(data), initial)
        # truth/model/error stacked along the height dim (dim=2): 3 * H.
        B = min(data["action"].shape[0], 6)
        H = data["image"].shape[2]
        assert out.shape[0] == B
        assert out.shape[1] == T
        assert out.shape[2] == 3 * H

    def test_does_not_require_grad(self):
        # video_pred is @torch.no_grad.
        agent, gs = make_real_dreamer(model__rep_loss="dreamer")
        data, initial = _sample(agent, gs)
        out = agent.video_pred(dict(data), initial)
        assert not out.requires_grad


class TestVideoPredUnsupported:
    @pytest.mark.parametrize("rep_loss", ["r2dreamer", "infonce", "dreamerpro"])
    def test_non_dreamer_raises(self, rep_loss):
        agent, gs = make_real_dreamer(model__rep_loss=rep_loss)
        data, initial = _sample(agent, gs)
        with pytest.raises(NotImplementedError):
            agent.video_pred(dict(data), initial)
