import torch
from torch import distributions as torchd

from her_dream.rssm import RSSM

from .conftest import A, B, D, E, K, S, T, make_rssm_config


class TestRSSMInit:
    def test_flat_stoch_attribute(self, rssm):
        assert rssm.flat_stoch == S * K

    def test_feat_size_attribute(self, rssm):
        assert rssm.feat_size == S * K + D

    def test_extra_obs_layers_add_modules(self):
        rssm_1 = RSSM(make_rssm_config(obs_layers=1), embed_size=E, act_dim=A)
        rssm_2 = RSSM(make_rssm_config(obs_layers=2), embed_size=E, act_dim=A)
        assert len(list(rssm_2._obs_net.children())) > len(list(rssm_1._obs_net.children()))

    def test_extra_img_layers_add_modules(self):
        rssm_1 = RSSM(make_rssm_config(img_layers=1), embed_size=E, act_dim=A)
        rssm_2 = RSSM(make_rssm_config(img_layers=2), embed_size=E, act_dim=A)
        assert len(list(rssm_2._img_net.children())) > len(list(rssm_1._img_net.children()))


class TestInitial:
    def test_stoch_shape(self, rssm):
        stoch, _ = rssm.initial(B)
        assert stoch.shape == (B, S, K)

    def test_deter_shape(self, rssm):
        _, deter = rssm.initial(B)
        assert deter.shape == (B, D)

    def test_stoch_all_zeros(self, rssm):
        stoch, _ = rssm.initial(B)
        assert stoch.sum().item() == 0.0

    def test_deter_all_zeros(self, rssm):
        _, deter = rssm.initial(B)
        assert deter.sum().item() == 0.0


class TestObsStep:
    def test_stoch_output_shape(self, rssm, stoch, deter, action, embed, reset):
        stoch_out, _, _ = rssm.obs_step(stoch, deter, action, embed, reset)
        assert stoch_out.shape == (B, S, K)

    def test_deter_output_shape(self, rssm, stoch, deter, action, embed, reset):
        _, deter_out, _ = rssm.obs_step(stoch, deter, action, embed, reset)
        assert deter_out.shape == (B, D)

    def test_logit_output_shape(self, rssm, stoch, deter, action, embed, reset):
        _, _, logit_out = rssm.obs_step(stoch, deter, action, embed, reset)
        assert logit_out.shape == (B, S, K)

    def test_reset_zeroes_prior_state(self, rssm, stoch, deter, action, embed):
        # reset=True must zero stoch, deter, and action before the transition.
        # Compare the deterministic outputs (deter and logit — no stochastic sampling)
        # against a fresh step with explicit zero inputs and reset=False.
        all_reset = torch.ones(B, dtype=torch.bool)
        no_reset = torch.zeros(B, dtype=torch.bool)

        with torch.no_grad():
            _, deter_reset, logit_reset = rssm.obs_step(stoch, deter, action, embed, all_reset)
            _, deter_zero, logit_zero = rssm.obs_step(
                torch.zeros_like(stoch), torch.zeros_like(deter), torch.zeros_like(action), embed, no_reset
            )

        assert torch.allclose(deter_reset, deter_zero)
        assert torch.allclose(logit_reset, logit_zero)

    def test_gradient_flows(self, rssm, stoch, deter, action, embed, reset):
        stoch_out, deter_out, logit_out = rssm.obs_step(stoch, deter, action, embed, reset)
        (stoch_out.sum() + deter_out.sum() + logit_out.sum()).backward()
        first_linear = list(rssm._obs_net.children())[0]
        assert first_linear.weight.grad is not None


class TestObserve:
    def test_stochs_shape(self, rssm, embed_seq, action_seq, initial, reset_seq):
        stochs, _, _ = rssm.observe(embed_seq, action_seq, initial, reset_seq)
        assert stochs.shape == (B, T, S, K)

    def test_deters_shape(self, rssm, embed_seq, action_seq, initial, reset_seq):
        _, deters, _ = rssm.observe(embed_seq, action_seq, initial, reset_seq)
        assert deters.shape == (B, T, D)

    def test_logits_shape(self, rssm, embed_seq, action_seq, initial, reset_seq):
        _, _, logits = rssm.observe(embed_seq, action_seq, initial, reset_seq)
        assert logits.shape == (B, T, S, K)


class TestImgStep:
    def test_stoch_shape(self, rssm, stoch, deter, action):
        stoch_out, _, _ = rssm.img_step(stoch, deter, action)
        assert stoch_out.shape == (B, S, K)

    def test_deter_shape(self, rssm, stoch, deter, action):
        _, deter_out, _ = rssm.img_step(stoch, deter, action)
        assert deter_out.shape == (B, D)

    def test_logit_shape(self, rssm, stoch, deter, action):
        _, _, logit_out = rssm.img_step(stoch, deter, action)
        assert logit_out.shape == (B, S, K)


class TestPrior:
    def test_stoch_shape(self, rssm, deter):
        stoch_out, _ = rssm.prior(deter)
        assert stoch_out.shape == (B, S, K)

    def test_logit_shape(self, rssm, deter):
        _, logit_out = rssm.prior(deter)
        assert logit_out.shape == (B, S, K)


class TestImagineWithAction:
    def test_stochs_shape(self, rssm, stoch, deter, action_seq):
        stochs, _ = rssm.imagine_with_action(stoch, deter, action_seq)
        assert stochs.shape == (B, T, S, K)

    def test_deters_shape(self, rssm, stoch, deter, action_seq):
        _, deters = rssm.imagine_with_action(stoch, deter, action_seq)
        assert deters.shape == (B, T, D)


class TestGetFeat:
    def test_feat_shape(self, rssm, stoch, deter):
        feat = rssm.get_feat(stoch, deter)
        assert feat.shape == (B, S * K + D)

    def test_stoch_portion_matches_flat(self, rssm, stoch, deter):
        feat = rssm.get_feat(stoch, deter)
        assert torch.equal(feat[:, : S * K], stoch.reshape(B, S * K))

    def test_deter_portion_matches(self, rssm, stoch, deter):
        feat = rssm.get_feat(stoch, deter)
        assert torch.equal(feat[:, S * K :], deter)


class TestGetDist:
    def test_returns_independent_distribution(self, rssm, logit):
        dist = rssm.get_dist(logit)
        assert isinstance(dist, torchd.independent.Independent)

    def test_sample_shape(self, rssm, logit):
        dist = rssm.get_dist(logit)
        assert dist.rsample().shape == (B, S, K)


class TestKLLoss:
    def test_dyn_loss_shape(self, rssm, post_logit, prior_logit):
        dyn, _ = rssm.kl_loss(post_logit, prior_logit, free=1.0)
        assert dyn.shape == (B, T)

    def test_rep_loss_shape(self, rssm, post_logit, prior_logit):
        _, rep = rssm.kl_loss(post_logit, prior_logit, free=1.0)
        assert rep.shape == (B, T)

    def test_free_bits_clamp(self, rssm):
        free = 10.0
        identical = torch.zeros(B, T, S, K)
        dyn, rep = rssm.kl_loss(identical, identical, free=free)
        assert (dyn >= free).all()
        assert (rep >= free).all()

    def test_rep_gradient_flows_through_post(self, rssm, post_logit, prior_logit):
        post = post_logit.detach().requires_grad_(True)
        prior = prior_logit.detach().requires_grad_(True)
        _, rep = rssm.kl_loss(post, prior, free=0.0)
        rep.sum().backward()
        assert post.grad is not None
        assert prior.grad is None  # prior is detached inside rep_loss

    def test_dyn_gradient_flows_through_prior(self, rssm, post_logit, prior_logit):
        post = post_logit.detach().requires_grad_(True)
        prior = prior_logit.detach().requires_grad_(True)
        dyn, _ = rssm.kl_loss(post, prior, free=0.0)
        dyn.sum().backward()
        assert prior.grad is not None
        assert post.grad is None  # post is detached inside dyn_loss
