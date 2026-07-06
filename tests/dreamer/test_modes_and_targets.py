"""Tests for freeze/distill wiring and the slow-target / clone machinery.

Covers `_apply_freeze_wm`, `_apply_train_text_only`, `to()` re-cloning,
`clone_and_freeze`, `_update_slow_target`, and `train()` keeping the slow value
network in eval mode.
"""

import pytest
import torch

from tests.dreamer.conftest import make_real_dreamer


def _all_frozen(module):
    return all(not p.requires_grad for p in module.parameters())


def _any_trainable(module):
    return any(p.requires_grad for p in module.parameters())


class TestApplyFreezeWM:
    def test_freezes_encoder_and_rssm(self):
        agent, _ = make_real_dreamer(freeze_wm=True)
        assert _all_frozen(agent.encoder)
        assert _all_frozen(agent.rssm)

    def test_actor_value_still_trainable(self):
        agent, _ = make_real_dreamer(freeze_wm=True)
        assert _any_trainable(agent.actor)
        assert _any_trainable(agent.value)

    def test_freezes_decoder_for_dreamer(self):
        agent, _ = make_real_dreamer(freeze_wm=True, model__rep_loss="dreamer")
        assert _all_frozen(agent.decoder)

    @pytest.mark.parametrize("rep_loss", ["r2dreamer", "infonce"])
    def test_freezes_projector(self, rep_loss):
        agent, _ = make_real_dreamer(freeze_wm=True, model__rep_loss=rep_loss)
        assert _all_frozen(agent.prj)

    def test_freezes_dreamerpro_modules(self):
        agent, _ = make_real_dreamer(freeze_wm=True, model__rep_loss="dreamerpro")
        assert not agent._prototypes.requires_grad
        assert _all_frozen(agent.obs_proj)
        assert _all_frozen(agent.feat_proj)

    def test_freezes_text_encoder(self):
        agent, _ = make_real_dreamer(freeze_wm=True, mission_text=True)
        assert _all_frozen(agent.text_encoder)


class TestApplyTrainTextOnly:
    def test_freezes_everything_but_text_encoder(self):
        agent, _ = make_real_dreamer(train_text_only=True, mission_text=True)
        assert _all_frozen(agent.encoder)
        assert _all_frozen(agent.rssm)
        assert _all_frozen(agent.actor)
        assert _all_frozen(agent.value)

    def test_text_encoder_stays_trainable(self):
        agent, _ = make_real_dreamer(train_text_only=True, mission_text=True)
        assert _any_trainable(agent.text_encoder)


class TestTo:
    def test_to_reclones_normal_agent(self, default_dreamer):
        default_dreamer.to("cpu")
        # frozen clones exist and are non-trainable after moving.
        assert _all_frozen(default_dreamer._frozen_encoder)
        assert _all_frozen(default_dreamer._frozen_rssm)

    def test_to_reapplies_freeze_wm(self):
        agent, _ = make_real_dreamer(freeze_wm=True)
        agent.to("cpu")
        assert _all_frozen(agent.encoder)

    def test_to_reapplies_train_text_only(self):
        agent, _ = make_real_dreamer(train_text_only=True, mission_text=True)
        agent.to("cpu")
        assert _all_frozen(agent.actor)
        assert _any_trainable(agent.text_encoder)


class TestCloneAndFreeze:
    def test_frozen_clones_share_data_and_are_frozen(self, default_dreamer):
        for live, frozen in zip(default_dreamer.encoder.parameters(), default_dreamer._frozen_encoder.parameters()):
            # requires_grad disabled on the frozen clone, data shared with live.
            assert not frozen.requires_grad
            assert frozen.data.data_ptr() == live.data.data_ptr()

    def test_all_frozen_networks_present(self, default_dreamer):
        for name in ("_frozen_encoder", "_frozen_rssm", "_frozen_actor", "_frozen_value", "_frozen_slow_value"):
            assert hasattr(default_dreamer, name)


class TestSlowTarget:
    def test_slow_value_frozen(self, default_dreamer):
        assert _all_frozen(default_dreamer._slow_value)

    def test_update_slow_target_mixes_on_schedule(self, default_dreamer):
        agent = default_dreamer
        # Make the live value params distinct so a copy is observable.
        with torch.no_grad():
            for p in agent.value.parameters():
                p.add_(1.0)
        before = [s.data.clone() for s in agent._slow_value.parameters()]
        agent._update_slow_target()  # _slow_value_updates starts at 0 -> mixes
        moved = any(not torch.equal(b, s.data) for b, s in zip(before, agent._slow_value.parameters()))
        assert moved

    def test_update_slow_target_skips_off_schedule(self):
        agent, _ = make_real_dreamer(model__slow_target_update=3)
        # step the counter to 1 (not a multiple of 3) -> no mix this call.
        agent._slow_value_updates = 1
        with torch.no_grad():
            for p in agent.value.parameters():
                p.add_(1.0)
        before = [s.data.clone() for s in agent._slow_value.parameters()]
        agent._update_slow_target()
        unchanged = all(torch.equal(b, s.data) for b, s in zip(before, agent._slow_value.parameters()))
        assert unchanged


class TestTrainMode:
    def test_slow_value_stays_eval_after_train(self, default_dreamer):
        default_dreamer.train()
        assert default_dreamer._slow_value.training is False

    def test_train_returns_self(self, default_dreamer):
        assert default_dreamer.train() is default_dreamer
