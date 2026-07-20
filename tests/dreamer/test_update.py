"""Tests for `Dreamer.update` / `_cal_grad` — the full optimization step.

These drive a real (tiny) `Dreamer` for one optimization step against a
`StubReplayBuffer`, covering:
- every representation-loss branch (r2dreamer / dreamer / infonce / dreamerpro),
- the special training modes (mission_text, wm_only, freeze_wm, train_text_only),
- every `goal_type` (which selects the reward input and, for log_prob, the
  threshold one-hot concatenation),
- the `log_grads` logging path,
- both `random_translate` branches used by dreamerpro augmentation,
- and the unknown-`rep_loss` NotImplementedError.

Losses are intentionally *not* asserted finite: under fp16 autocast the barlow
loss can be NaN, in which case GradScaler simply skips the optimizer step. We
assert the returned metrics structure and the replay write-back instead.
"""

import pytest

from tests.dreamer.conftest import StubReplayBuffer, make_real_dreamer


def _run_once(agent, goal_shape, mission=False):
    buf = StubReplayBuffer(goal_shape, mission=mission, act_dim=agent.act_dim)
    metrics = agent.update(buf)
    return metrics, buf


class TestRepLossModes:
    @pytest.mark.parametrize("rep_loss", ["r2dreamer", "dreamer", "infonce", "dreamerpro"])
    def test_update_runs_and_returns_metrics(self, rep_loss):
        agent, gs = make_real_dreamer(model__rep_loss=rep_loss)
        metrics, _ = _run_once(agent, gs)
        assert "opt/loss" in metrics
        assert "loss/dyn" in metrics
        assert "loss/rep" in metrics

    def test_rep_specific_loss_key_present(self):
        # Each rep loss contributes its own named loss term.
        for rep_loss, key in [
            ("r2dreamer", "loss/barlow"),
            ("infonce", "loss/infonce"),
        ]:
            agent, gs = make_real_dreamer(model__rep_loss=rep_loss)
            metrics, _ = _run_once(agent, gs)
            assert key in metrics

    def test_dreamer_recon_losses_present(self):
        agent, gs = make_real_dreamer(model__rep_loss="dreamer")
        metrics, _ = _run_once(agent, gs)
        # decoder outputs each become a loss term keyed by the obs name.
        assert any(k.startswith("loss/") and k.split("/", 1)[1] in agent.decoder.all_keys for k in metrics)

    def test_dreamerpro_swav_temp_norm_present(self):
        agent, gs = make_real_dreamer(model__rep_loss="dreamerpro")
        metrics, _ = _run_once(agent, gs)
        for key in ("loss/swav", "loss/temp", "loss/norm"):
            assert key in metrics

    def test_unknown_rep_loss_raises(self):
        # __init__ tolerates an unknown rep_loss (no aux module), but _cal_grad
        # falls through to `raise NotImplementedError`.
        agent, gs = make_real_dreamer(model__rep_loss="foobar")
        with pytest.raises(NotImplementedError):
            _run_once(agent, gs)


class TestWriteBack:
    def test_replay_update_called_with_detached_latents(self, default_dreamer, default_buffer):
        default_dreamer.update(default_buffer)
        assert len(default_buffer.update_calls) == 1
        _, stoch, deter = default_buffer.update_calls[0]
        assert not stoch.requires_grad
        assert not deter.requires_grad


class TestTrainingModes:
    def test_mission_text_adds_text_kl(self):
        agent, gs = make_real_dreamer(mission_text=True)
        metrics, _ = _run_once(agent, gs, mission=True)
        assert "loss/text_kl" in metrics

    def test_wm_only_skips_actor_critic(self):
        # wm_only returns before the imagination rollout: no policy/value losses.
        agent, gs = make_real_dreamer(wm_only=True)
        metrics, _ = _run_once(agent, gs)
        assert "loss/policy" not in metrics
        assert "loss/value" not in metrics
        assert "loss/dyn" in metrics

    def test_freeze_wm_skips_wm_losses_but_trains_actor_critic(self):
        agent, gs = make_real_dreamer(freeze_wm=True)
        metrics, _ = _run_once(agent, gs)
        # WM/KL losses are skipped; actor-critic still runs.
        assert "loss/dyn" not in metrics
        assert "loss/policy" in metrics
        assert "loss/repval" in metrics

    def test_train_text_only_only_text_kl(self):
        agent, gs = make_real_dreamer(train_text_only=True, mission_text=True)
        metrics, _ = _run_once(agent, gs, mission=True)
        assert "loss/text_kl" in metrics
        assert "loss/policy" not in metrics
        assert "loss/dyn" not in metrics


class TestGoalTypes:
    @pytest.mark.parametrize("goal_type", ["full", "first_row", "row_by_row", "argmax_full", "log_prob", "prob"])
    def test_update_runs_for_each_goal_type(self, goal_type):
        agent, gs = make_real_dreamer(goal_type=goal_type)
        metrics, _ = _run_once(agent, gs)
        assert "opt/loss" in metrics
        assert "loss/policy" in metrics


class TestLogGrads:
    def test_log_grads_adds_grad_and_update_metrics(self):
        agent, gs = make_real_dreamer(model__log_grads=True)
        metrics, _ = _run_once(agent, gs)
        for key in ("opt/grad_norm", "opt/grad_rms", "opt/param_rms", "opt/update_rms"):
            assert key in metrics


class TestDreamerProAugmentation:
    @pytest.mark.parametrize("same_across_time", [True, False])
    @pytest.mark.parametrize("bilinear", [True, False])
    def test_augmentation_variants_run(self, same_across_time, bilinear):
        # Covers both branches of random_translate (shift shape, sampling mode).
        agent, gs = make_real_dreamer(
            model__rep_loss="dreamerpro",
            **{
                "model.dreamer_pro.aug.same_across_time": same_across_time,
                "model.dreamer_pro.aug.bilinear": bilinear,
            },
        )
        metrics, _ = _run_once(agent, gs)
        assert "loss/swav" in metrics


class TestOptimizerMetrics:
    def test_reports_lr_and_grad_scale(self, default_dreamer, default_buffer):
        metrics = default_dreamer.update(default_buffer)
        assert "opt/lr" in metrics
        assert "opt/lr_last" in metrics
        assert "opt/grad_scale" in metrics

    def test_scheduler_advances_across_updates(self, default_dreamer, default_buffer):
        first = default_dreamer.update(default_buffer)["opt/lr_last"]
        second = default_dreamer.update(default_buffer)["opt/lr_last"]
        # warmup=1000 by default, so lr strictly increases early on.
        assert second > first

    def test_no_warmup_keeps_constant_lr(self):
        # warmup=0 (falsy) takes the lr_lambda `return 1.0` branch: flat lr.
        agent, gs = make_real_dreamer(model__warmup=0)
        buf = StubReplayBuffer(gs, act_dim=agent.act_dim)
        first = agent.update(buf)["opt/lr_last"]
        second = agent.update(buf)["opt/lr_last"]
        assert first == second
