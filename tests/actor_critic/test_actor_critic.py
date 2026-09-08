"""Tests for `ActorCritic`: input layout, objectives, naming and lifecycle."""

import pytest
import torch

from her_dream.actor_critic import ActorCritic
from tests.actor_critic.conftest import T_IMAG, A, B, F, G, make_ac
from tests.dreamer.conftest import act_cont, act_discrete, act_multi, act_with_n, build_model_config


def _mlphead_in_features(head):
    return next(m for m in head.modules() if isinstance(m, torch.nn.Linear)).in_features


class TestConstruction:
    def test_actor_and_critic_see_feat_plus_goal(self, ac):
        assert _mlphead_in_features(ac.actor) == F + G
        assert _mlphead_in_features(ac.value) == F + G

    def test_log_prob_appends_a_threshold_onehot(self):
        ac = make_ac(goal_type="log_prob")
        assert ac.threshold_bins > 0
        assert ac.threshold_onehot.sum().item() == pytest.approx(1.0)
        assert _mlphead_in_features(ac.actor) == F + G + ac.threshold_bins

    def test_other_goal_types_have_no_threshold(self, ac):
        assert ac.threshold_bins == 0
        assert not hasattr(ac, "threshold_onehot")

    @pytest.mark.parametrize(
        "act_space, discrete",
        [(act_multi((2, 3)), True), (act_cont(A), False), (act_with_n(A), True)],
    )
    def test_action_space_selects_the_actor_dist(self, act_space, discrete):
        # `discrete` here means "the actor emits (multi-)onehot", i.e. not the
        # continuous bounded_normal branch. `act_with_n` exposes `.n` only.
        ac = make_ac(act=act_space)
        assert ac.act_discrete is (discrete and not hasattr(act_space, "n"))
        assert ac.actor(torch.randn(B, ac.input_size)).rsample().shape == (B, ac.act_dim)

    def test_slow_value_starts_frozen_and_matches_the_critic(self, ac):
        assert all(not p.requires_grad for p in ac._slow_value.parameters())
        for v, s in zip(ac.value.parameters(), ac._slow_value.parameters()):
            assert torch.equal(v.data, s.data)


class TestPolicyInput:
    def test_concatenates_feat_and_flat_goal(self, ac):
        feat, goal = torch.randn(B, F), torch.randn(B, G)
        out = ac.policy_input(feat, goal)
        assert out.shape == (B, F + G)
        assert torch.equal(out[:, :F], feat)
        assert torch.equal(out[:, F:], goal)

    def test_flattens_a_grouped_goal(self, ac):
        # A (S, K) goal is flattened, matching how the buffer stores it.
        feat, goal = torch.randn(B, F), torch.randn(B, 4, 4)
        assert torch.equal(ac.policy_input(feat, goal)[:, F:], goal.reshape(B, G))

    def test_handles_a_time_dimension(self, ac):
        feat, goal = torch.randn(B, T_IMAG, F), torch.randn(B, T_IMAG, G)
        assert ac.policy_input(feat, goal).shape == (B, T_IMAG, F + G)

    def test_appends_the_threshold_onehot_last(self):
        ac = make_ac(goal_type="log_prob")
        feat, goal = torch.randn(B, F), torch.randn(B, G)
        out = ac.policy_input(feat, goal)
        assert out.shape == (B, F + G + ac.threshold_bins)
        assert torch.equal(out[:, F + G :], ac.threshold_onehot.expand(B, -1))

    def test_policy_and_frozen_policy_agree_at_init(self, ac):
        feat, goal = torch.randn(B, F), torch.randn(B, G)
        assert torch.allclose(ac.policy(feat, goal).mode, ac.frozen_policy(feat, goal).mode)


class TestLambdaReturn:
    def test_zero_lambda_is_the_one_step_return(self, ac):
        shape = (B, T_IMAG, 1)
        last, term = torch.zeros(shape), torch.zeros(shape)
        reward, value = torch.full(shape, 2.0), torch.full(shape, 5.0)
        ret = ac.lambda_return(last, term, reward, value, value, disc=1.0, lamb=0.0)
        # lamb=0 -> reward_t + disc * boot_t at every step.
        assert torch.allclose(ret, torch.full((B, T_IMAG - 1, 1), 7.0))

    def test_termination_cuts_the_bootstrap(self, ac):
        shape = (B, T_IMAG, 1)
        last, term = torch.zeros(shape), torch.ones(shape)
        reward, value = torch.full(shape, 2.0), torch.full(shape, 5.0)
        ret = ac.lambda_return(last, term, reward, value, value, disc=1.0, lamb=0.0)
        assert torch.allclose(ret, torch.full((B, T_IMAG - 1, 1), 2.0))

    def test_shape_mismatch_asserts(self, ac):
        shape = (B, T_IMAG, 1)
        ok = torch.zeros(shape)
        with pytest.raises(AssertionError):
            ac.lambda_return(ok, ok, ok, ok, torch.zeros(B, T_IMAG + 1, 1), 1.0, 0.5)


class TestImaginationLoss:
    def test_returns_policy_and_value_losses(self, ac, imag_batch):
        losses, _, _ = ac.imagination_loss(
            imag_batch.feat, imag_batch.action, imag_batch.reward, imag_batch.cont, imag_batch.goal
        )
        assert set(losses) == {"policy", "value"}
        assert all(v.ndim == 0 for v in losses.values())

    def test_returns_the_lambda_return_for_bootstrapping(self, ac, imag_batch):
        _, _, ret = ac.imagination_loss(
            imag_batch.feat, imag_batch.action, imag_batch.reward, imag_batch.cont, imag_batch.goal
        )
        assert ret.shape == (B, T_IMAG - 1, 1)

    def test_reports_the_expected_metrics(self, ac, imag_batch):
        _, metrics, _ = ac.imagination_loss(
            imag_batch.feat, imag_batch.action, imag_batch.reward, imag_batch.cont, imag_batch.goal
        )
        assert {"ret", "adv", "con", "rew", "val", "weight", "action_entropy"} <= set(metrics)
        assert metrics["con"].item() == pytest.approx(1.0)
        assert metrics["rew"].item() == pytest.approx(-1.0)

    def test_gradients_reach_the_actor_and_the_critic(self, ac, imag_batch):
        losses, _, _ = ac.imagination_loss(
            imag_batch.feat, imag_batch.action, imag_batch.reward, imag_batch.cont, imag_batch.goal
        )
        sum(losses.values()).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ac.actor.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ac.value.parameters())

    def test_continuation_below_one_discounts_the_weight(self, ac, imag_batch):
        _, full, _ = ac.imagination_loss(
            imag_batch.feat, imag_batch.action, imag_batch.reward, imag_batch.cont, imag_batch.goal
        )
        _, half, _ = ac.imagination_loss(
            imag_batch.feat, imag_batch.action, imag_batch.reward, 0.5 * imag_batch.cont, imag_batch.goal
        )
        assert half["weight"].item() < full["weight"].item()


class TestReplayValueLoss:
    def test_returns_the_repval_loss(self, ac, replay_batch):
        losses, _ = ac.replay_value_loss(
            replay_batch.feat,
            replay_batch.goal,
            replay_batch.last,
            replay_batch.term,
            replay_batch.reward,
            replay_batch.boot,
        )
        assert set(losses) == {"repval"}
        assert losses["repval"].ndim == 0

    def test_gradient_flows_back_into_the_features(self, ac, replay_batch):
        # This is what lets the critic train the world model. The critic head is
        # built with outscale=0, so its last layer starts at exactly zero and
        # passes a zero gradient back; nudge it off zero as one update would.
        with torch.no_grad():
            for p in ac.value.parameters():
                p.add_(0.1)
        losses, _ = ac.replay_value_loss(
            replay_batch.feat,
            replay_batch.goal,
            replay_batch.last,
            replay_batch.term,
            replay_batch.reward,
            replay_batch.boot,
        )
        losses["repval"].backward()
        assert replay_batch.feat.grad is not None
        assert replay_batch.feat.grad.abs().sum() > 0

    def test_reports_replay_tensorstats(self, ac, replay_batch):
        _, metrics = ac.replay_value_loss(
            replay_batch.feat,
            replay_batch.goal,
            replay_batch.last,
            replay_batch.term,
            replay_batch.reward,
            replay_batch.boot,
        )
        assert {"ret_replay_mean", "value_replay_mean", "slow_value_replay_mean"} <= set(metrics)


class TestNaming:
    def test_unnamed_instance_keeps_the_bare_keys(self, ac):
        assert ac.key("policy") == "policy"
        assert set(ac.optim_modules()) == {"actor", "value"}
        assert ac.loss_scales({"policy": 1.0, "value": 1.0, "repval": 0.3}) == {}

    def test_named_instance_prefixes_every_key(self):
        ac = make_ac(name="explore")
        assert ac.key("policy") == "explore_policy"
        assert set(ac.optim_modules()) == {"explore_actor", "explore_value"}

    def test_named_instance_mirrors_the_base_loss_scales(self):
        ac = make_ac(name="explore")
        base = {"policy": 1.0, "value": 1.0, "repval": 0.3}
        assert ac.loss_scales(base) == {
            "explore_policy": 1.0,
            "explore_value": 1.0,
            "explore_repval": 0.3,
        }

    def test_named_instance_prefixes_losses_and_metrics(self, imag_batch, replay_batch):
        ac = make_ac(name="explore")
        losses, metrics, _ = ac.imagination_loss(
            imag_batch.feat, imag_batch.action, imag_batch.reward, imag_batch.cont, imag_batch.goal
        )
        assert set(losses) == {"explore_policy", "explore_value"}
        assert "explore_ret" in metrics and "ret" not in metrics
        replay_losses, _ = ac.replay_value_loss(
            replay_batch.feat,
            replay_batch.goal,
            replay_batch.last,
            replay_batch.term,
            replay_batch.reward,
            replay_batch.boot,
        )
        assert set(replay_losses) == {"explore_repval"}

    def test_two_instances_do_not_collide(self):
        task, explore = make_ac(), make_ac(name="explore")
        assert not set(task.optim_modules()) & set(explore.optim_modules())


class TestLifecycle:
    def test_clone_and_freeze_shares_storage(self, ac):
        for live, frozen in zip(ac.actor.parameters(), ac._frozen_actor.parameters()):
            assert not frozen.requires_grad
            assert frozen.data.data_ptr() == live.data.data_ptr()

    def test_all_three_clones_exist(self, ac):
        for name in ("_frozen_actor", "_frozen_value", "_frozen_slow_value"):
            assert hasattr(ac, name)

    def test_update_slow_target_mixes_on_schedule(self, ac):
        with torch.no_grad():
            for p in ac.value.parameters():
                p.add_(1.0)
        before = [s.data.clone() for s in ac._slow_value.parameters()]
        ac.update_slow_target()
        assert any(not torch.equal(b, s.data) for b, s in zip(before, ac._slow_value.parameters()))

    def test_update_slow_target_skips_off_schedule(self):
        ac = make_ac(model__slow_target_update=3)
        ac._slow_value_updates = 1
        with torch.no_grad():
            for p in ac.value.parameters():
                p.add_(1.0)
        before = [s.data.clone() for s in ac._slow_value.parameters()]
        ac.update_slow_target()
        assert all(torch.equal(b, s.data) for b, s in zip(before, ac._slow_value.parameters()))

    def test_freeze_disables_actor_and_critic_grads(self, ac):
        ac.freeze()
        assert all(not p.requires_grad for p in ac.actor.parameters())
        assert all(not p.requires_grad for p in ac.value.parameters())

    def test_train_keeps_the_slow_value_in_eval(self, ac):
        ac.train()
        assert ac._slow_value.training is False
        assert ac.actor.training is True

    def test_train_from_a_parent_keeps_the_slow_value_in_eval(self, ac):
        # Dreamer.train() recurses into children; the override must survive that.
        parent = torch.nn.Module()
        parent.ac = ac
        parent.train()
        assert ac._slow_value.training is False

    def test_slow_value_is_absent_from_the_optimizer_modules(self, ac):
        assert ac._slow_value not in ac.optim_modules().values()


class TestGoalAgnostic:
    """`goal_size=0` — the layout an exploration policy uses."""

    def test_policy_input_is_the_bare_feature(self):
        ac = make_ac(goal_size=0)
        feat = torch.randn(B, F)
        assert torch.equal(ac.policy_input(feat, None), feat)

    def test_actor_and_critic_take_the_feature_alone(self):
        ac = make_ac(goal_size=0)
        assert _mlphead_in_features(ac.actor) == F
        assert _mlphead_in_features(ac.value) == F

    def test_the_threshold_onehot_is_suppressed(self):
        # The threshold describes the goal acceptance criterion, so a policy that
        # ignores goals takes neither — even under log_prob.
        ac = make_ac(goal_type="log_prob", goal_size=0)
        assert ac.threshold_bins == 0
        assert not hasattr(ac, "threshold_onehot")

    def test_imagination_loss_accepts_no_goal(self, imag_batch):
        ac = make_ac(goal_size=0)
        losses, metrics, ret = ac.imagination_loss(
            imag_batch.feat, imag_batch.action, imag_batch.reward, imag_batch.cont, None
        )
        assert set(losses) == {"policy", "value"}
        assert ret.shape == (B, T_IMAG - 1, 1)

    def test_replay_value_loss_accepts_no_goal(self, replay_batch):
        ac = make_ac(goal_size=0)
        losses, _ = ac.replay_value_loss(
            replay_batch.feat, None, replay_batch.last, replay_batch.term, replay_batch.reward, replay_batch.boot
        )
        assert set(losses) == {"repval"}


class TestSharedConfigNode:
    """Two instances from one config — how Dreamer builds task + explorer."""

    @pytest.mark.parametrize("act", ["discrete", "multi", "cont", "n"])
    def test_a_second_instance_reuses_the_collapsed_dist(self, act):
        # __init__ collapses config.actor.dist in place; the second build must
        # not look for `.disc` on the already-collapsed node.
        cfg = build_model_config()
        act_space = {"discrete": act_discrete, "multi": act_multi, "cont": act_cont, "n": act_with_n}[act]()
        first = ActorCritic(cfg.model, F, act_space, G)
        second = ActorCritic(cfg.model, F, act_space, 0, name="explore")
        assert first.act_discrete == second.act_discrete
        assert _mlphead_in_features(second.actor) == F

    def test_the_two_instances_have_separate_parameters(self):
        cfg = build_model_config()
        task = ActorCritic(cfg.model, F, act_discrete(A), G)
        explorer = ActorCritic(cfg.model, F, act_discrete(A), 0, name="explore")
        task_ids = {id(p) for p in task.parameters()}
        assert not task_ids & {id(p) for p in explorer.parameters()}
