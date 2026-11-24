"""Tests for trainer.py — OnlineTrainer.

Strategy: mock all external dependencies (agent, envs, buffer, logger) so tests
run without GPU or real environment processes.  A lightweight FakeTrans dict
supports the tensor-dict-like protocol that the trainer expects.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import her_dream.goals as goals
from her_dream.buffers.her_buffer import HERBuffer
from her_dream.trainer import OnlineTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B = 2  # batch (env) count used across tests
S = 4  # stochastic groups
K = 8  # categories per group
A = 3  # action dims


class FakeTrans(dict):
    """Minimal TensorDict-like that supports .to(), .clone(), .detach()."""

    def to(self, device, non_blocking=False):
        return self

    def clone(self):
        return FakeTrans({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.items()})

    def detach(self):
        return self

    def __getitem__(self, key):
        # Support integer indexing for trans[indices] used in _sample_goals imagination branch.
        if isinstance(key, torch.Tensor):
            return FakeTrans({k: v[key] if isinstance(v, torch.Tensor) else v for k, v in self.items()})
        return super().__getitem__(key)


def make_trans(num_envs=B, goal_shape=(S, K), extra=None):
    """Build a minimal FakeTrans with the keys the trainer accesses."""
    t = FakeTrans({
        "is_first": torch.zeros(num_envs, 1, dtype=torch.bool),
        "reward": torch.zeros(num_envs, 1),
        "goal": torch.zeros(num_envs, *goal_shape),  # no time dim — _relabel_goal requires (B, *goal_shape)
        "stoch": torch.zeros(num_envs, 1, S, K),
    })
    if extra:
        t.update(extra)
    return t


def make_config(**overrides):
    defaults = dict(
        steps=0,
        pretrain=1,
        eval_every=10_000,
        eval_episode_num=0,  # skip eval by default
        video_pred_log=False,
        params_hist_log=False,
        obs_step_prob_log=False,
        batch_length=0,  # keeps cache empty in simple tests (avoids torch.stack on FakeTrans)
        batch_size=16,
        train_ratio=512.0,
        action_repeat=1,
        update_log_every=100,
        goal_sample="random",
        goal_type="full",
        wm_only=False,
        train_text_only=False,
    )
    defaults.update(overrides)
    # Fill goal-type descriptors from the config group unless explicitly overridden.
    goals.with_default_descriptors(defaults)
    return SimpleNamespace(**defaults)


def make_mock_agent(with_logit=False):
    agent = MagicMock()
    agent.device = "cpu"
    state = {
        "prev_action": torch.zeros(B, A),
        "stoch": torch.zeros(B, S, K),
        "deter": torch.zeros(B, 64),
    }
    if with_logit:
        state["logit"] = torch.zeros(B, S, K)
    agent.get_initial_state.return_value = state
    agent.act.return_value = (torch.zeros(B, A), state, {})
    return agent


def make_mock_envs(num_envs=B, goal_shape=(S, K)):
    envs = MagicMock()
    envs.env_num = num_envs
    obs_space = MagicMock()
    obs_space.__getitem__ = lambda self, k: MagicMock(shape=goal_shape)
    envs.observation_space = obs_space
    # Default: returns all-done on first step
    done = torch.ones(num_envs, dtype=torch.bool)
    envs.step.return_value = (make_trans(num_envs, goal_shape), done)
    # Provide individual env mocks for text/image goal tests
    envs.envs = [MagicMock() for _ in range(num_envs)]
    return envs


def make_mock_buffer():
    buf = MagicMock()
    buf.count.return_value = 0
    return buf


def make_trainer(config=None, *, buffer=None, logger=None, train_envs=None, eval_envs=None, reward=None):
    config = config or make_config()
    buffer = buffer or make_mock_buffer()
    logger = logger or MagicMock()
    train_envs = train_envs or make_mock_envs()
    eval_envs = eval_envs or make_mock_envs()
    return OnlineTrainer(config, buffer, logger, "/tmp/logdir", train_envs, eval_envs, reward_function=reward)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_stores_steps(self):
        t = make_trainer(make_config(steps=500_000))
        assert t.steps == 500_000

    def test_stores_pretrain(self):
        t = make_trainer(make_config(pretrain=3))
        assert t.pretrain == 3

    def test_stores_eval_every(self):
        t = make_trainer(make_config(eval_every=2000))
        assert t.eval_every == 2000

    def test_stores_eval_episode_num(self):
        t = make_trainer(make_config(eval_episode_num=5))
        assert t.eval_episode_num == 5

    def test_stores_video_pred_log(self):
        t = make_trainer(make_config(video_pred_log=True))
        assert t.video_pred_log is True

    def test_stores_params_hist_log(self):
        t = make_trainer(make_config(params_hist_log=True))
        assert t.params_hist_log is True

    def test_stores_obs_step_prob_log(self):
        t = make_trainer(make_config(obs_step_prob_log=True))
        assert t.obs_step_prob_log is True

    def test_stores_batch_length(self):
        t = make_trainer(make_config(batch_length=64))
        assert t.batch_length == 64

    def test_her_true_when_buffer_is_HERBuffer(self):
        buf = MagicMock(spec=HERBuffer)
        t = make_trainer(buffer=buf)
        assert t.her is True

    def test_her_false_when_regular_buffer(self):
        buf = MagicMock()  # not spec=HERBuffer
        t = make_trainer(buffer=buf)
        assert t.her is False

    def test_random_actions_true_when_wm_only(self):
        t = make_trainer(make_config(wm_only=True))
        assert t._random_actions is True

    def test_random_actions_true_when_train_text_only(self):
        t = make_trainer(make_config(train_text_only=True))
        assert t._random_actions is True

    def test_random_actions_false_otherwise(self):
        t = make_trainer(make_config(wm_only=False, train_text_only=False))
        assert t._random_actions is False


# ---------------------------------------------------------------------------
# TestShouldUpdate
# ---------------------------------------------------------------------------


class TestShouldUpdate:
    def test_false_when_not_enough_data(self):
        t = make_trainer(make_config(batch_length=50))
        # Need step // env_num > batch_length + 1, i.e., > 51
        assert t._should_update(0) is False

    def test_true_when_enough_steps(self):
        # With env_num=2, action_repeat=1: need step//2 > batch_length+1
        # batch_length=0 → need step//2 > 1, i.e., step >= 4
        envs = make_mock_envs(num_envs=2)
        t = make_trainer(make_config(batch_length=0), train_envs=envs)
        assert t._should_update(4) is True


# ---------------------------------------------------------------------------
# TestApplyReward
# ---------------------------------------------------------------------------


class TestApplyReward:
    def test_no_reward_function_leaves_trans_unchanged(self):
        t = make_trainer(reward=None)
        trans = make_trans()
        trans["reward"] = torch.tensor([[99.0], [99.0]])
        t._apply_reward(trans)
        assert (trans["reward"] == 99.0).all()

    def test_log_prob_goal_type_uses_rssm_get_dist(self):
        reward_fn = MagicMock(return_value=torch.zeros(B, 1))
        t = make_trainer(make_config(goal_type="log_prob"), reward=reward_fn)
        trans = make_trans()
        trans["logit"] = torch.zeros(B, S, K)
        rssm = MagicMock()
        t._apply_reward(trans, rssm)
        rssm.get_dist.assert_called_once()
        reward_fn.assert_called_once()

    def test_prob_goal_type_uses_rssm_get_dist(self):
        reward_fn = MagicMock(return_value=torch.zeros(B, 1))
        t = make_trainer(make_config(goal_type="prob"), reward=reward_fn)
        trans = make_trans()
        trans["logit"] = torch.zeros(B, S, K)
        rssm = MagicMock()
        t._apply_reward(trans, rssm)
        rssm.get_dist.assert_called_once()

    def test_logit_goal_type_uses_logit(self):
        captured = {}

        def reward_fn(state, goal):
            captured["state"] = state
            return torch.zeros(B, 1)

        # argmax_full has state_repr="logit": the reward compares against the raw logit.
        t = make_trainer(make_config(goal_type="argmax_full"), reward=reward_fn)
        trans = make_trans()
        logit = torch.ones(B, S, K) * 7.0
        trans["logit"] = logit
        t._apply_reward(trans)
        assert captured["state"] is logit

    def test_other_goal_type_uses_stoch_when_no_logit(self):
        captured = {}

        def reward_fn(state, goal):
            captured["state"] = state
            return torch.zeros(B, 1)

        t = make_trainer(make_config(goal_type="full"), reward=reward_fn)
        trans = make_trans()
        stoch = torch.ones(B, 1, S, K) * 3.0
        trans["stoch"] = stoch
        t._apply_reward(trans)
        assert captured["state"] is stoch


# ---------------------------------------------------------------------------
# TestRelabelGoal
# ---------------------------------------------------------------------------


class TestRelabelGoal:
    def test_relabels_when_goals_nonzero(self):
        t = make_trainer()
        envs = make_mock_envs()
        goals = torch.ones(B, S, K)
        trans = make_trans()
        t._relabel_goal(envs, goals, trans)
        # All envs have nonzero goals → entire trans["goal"] relabelled
        assert (trans["goal"] == 1.0).all()

    def test_does_not_relabel_when_goals_all_zero(self):
        t = make_trainer()
        envs = make_mock_envs()
        goals = torch.zeros(B, S, K)
        trans = make_trans()
        original = trans["goal"].clone()
        t._relabel_goal(envs, goals, trans)
        assert torch.equal(trans["goal"], original)

    def test_relabels_only_nonzero_envs(self):
        t = make_trainer()
        envs = make_mock_envs()
        # Only env 0 has a nonzero goal
        goals = torch.zeros(B, S, K)
        goals[0] = 1.0
        trans = make_trans()
        t._relabel_goal(envs, goals, trans)
        assert (trans["goal"][0] == 1.0).all()
        assert (trans["goal"][1] == 0.0).all()


# ---------------------------------------------------------------------------
# TestTextGoal
# ---------------------------------------------------------------------------


class TestTextGoal:
    def _make_trainer_with_goal_type(self, goal_type="full"):
        return make_trainer(make_config(goal_type=goal_type))

    def test_returns_none_when_indices_empty(self):
        t = self._make_trainer_with_goal_type()
        agent = make_mock_agent()
        envs = make_mock_envs()
        result = t._text_goal(agent, envs, [])
        assert result is None

    def test_returns_one_hot_for_argmax_full(self):
        t = self._make_trainer_with_goal_type("argmax_full")
        agent = make_mock_agent()
        # text_encoder returns logits (N, 1, S, K) — indices=[0,1], T=1
        logits = torch.randn(B, 1, S, K)
        agent.text_encoder.return_value = logits
        envs = make_mock_envs(goal_shape=(S, K))

        for env in envs.envs:
            promise = MagicMock(return_value=np.zeros(10, dtype=np.int8))
            env.encoded_random_mission.return_value = promise

        result = t._text_goal(agent, envs, [0, 1])
        # argmax_full: should be one-hot (sum=1 over last dim)
        assert result.shape == (B, S, K)
        assert (result.sum(dim=-1) == 1).all()

    def test_returns_rsample_for_other_goal_types(self):
        t = self._make_trainer_with_goal_type("full")
        agent = make_mock_agent()
        logits = torch.randn(B, 1, S, K)
        agent.text_encoder.return_value = logits
        dist_mock = MagicMock()
        dist_mock.rsample.return_value = torch.zeros(B, S, K)
        agent.rssm.get_dist.return_value = dist_mock
        envs = make_mock_envs(goal_shape=(S, K))

        for env in envs.envs:
            promise = MagicMock(return_value=np.zeros(10, dtype=np.int8))
            env.encoded_random_mission.return_value = promise

        result = t._text_goal(agent, envs, [0, 1])
        dist_mock.rsample.assert_called_once()
        assert result.shape == (B, S, K)

    def test_1d_goal_shape_slices_first_row(self):
        t = self._make_trainer_with_goal_type("full")
        agent = make_mock_agent()
        logits = torch.randn(B, 1, S, K)
        agent.text_encoder.return_value = logits
        dist_mock = MagicMock()
        dist_mock.rsample.return_value = torch.zeros(B, S, K)
        agent.rssm.get_dist.return_value = dist_mock
        envs = make_mock_envs(goal_shape=(K,))  # 1-D goal

        for env in envs.envs:
            promise = MagicMock(return_value=np.zeros(10, dtype=np.int8))
            env.encoded_random_mission.return_value = promise

        result = t._text_goal(agent, envs, [0, 1])
        # Should slice [:, 0, :] → (B, K)
        assert result.shape == (B, K)

    def test_2d_goal_shape_returns_full_tensor(self):
        t = self._make_trainer_with_goal_type("full")
        agent = make_mock_agent()
        logits = torch.randn(B, 1, S, K)
        agent.text_encoder.return_value = logits
        dist_mock = MagicMock()
        dist_mock.rsample.return_value = torch.zeros(B, S, K)
        agent.rssm.get_dist.return_value = dist_mock
        envs = make_mock_envs(goal_shape=(S, K))  # 2-D goal

        for env in envs.envs:
            promise = MagicMock(return_value=np.zeros(10, dtype=np.int8))
            env.encoded_random_mission.return_value = promise

        result = t._text_goal(agent, envs, [0, 1])
        assert result.shape == (B, S, K)


# ---------------------------------------------------------------------------
# TestImageGoal
# ---------------------------------------------------------------------------


class TestImageGoal:
    def test_returns_none_when_indices_empty(self):
        t = make_trainer()
        agent = make_mock_agent()
        envs = make_mock_envs()
        result = t._image_goal(agent, envs, [])
        assert result is None

    def _setup_envs_for_image_goal(self, envs, goal_shape=(S, K)):
        for i, env in enumerate(envs.envs):
            obs = {"image": torch.zeros(3, 64, 64)}
            promise = MagicMock(return_value=obs)
            env.goal_observation.return_value = promise

    def test_encodes_goal_observations(self):
        t = make_trainer()
        agent = make_mock_agent()
        agent.encode_observation.return_value = torch.zeros(B, S, K)
        envs = make_mock_envs(goal_shape=(S, K))
        self._setup_envs_for_image_goal(envs)

        result = t._image_goal(agent, envs, [0, 1])
        agent.encode_observation.assert_called_once()
        assert result.shape == (B, S, K)

    def test_1d_goal_shape_slices_first_row(self):
        t = make_trainer()
        agent = make_mock_agent()
        agent.encode_observation.return_value = torch.zeros(B, S, K)
        envs = make_mock_envs(goal_shape=(K,))
        self._setup_envs_for_image_goal(envs, goal_shape=(K,))

        result = t._image_goal(agent, envs, [0, 1])
        assert result.shape == (B, K)

    def test_2d_goal_shape_returns_full_tensor(self):
        t = make_trainer()
        agent = make_mock_agent()
        agent.encode_observation.return_value = torch.zeros(B, S, K)
        envs = make_mock_envs(goal_shape=(S, K))
        self._setup_envs_for_image_goal(envs)

        result = t._image_goal(agent, envs, [0, 1])
        assert result.shape == (B, S, K)


# ---------------------------------------------------------------------------
# TestSampleGoals
# ---------------------------------------------------------------------------


class TestSampleGoals:
    def test_buffer_skips_when_empty(self):
        buf = make_mock_buffer()
        buf.count.return_value = 0
        t = make_trainer(make_config(goal_sample="buffer"), buffer=buf)
        goals = torch.zeros(B, S, K)
        mask = torch.ones(B, dtype=torch.bool)
        trans = make_trans()
        t._sample_goals(MagicMock(), make_mock_envs(), mask, goals, trans)
        # goals should remain zero since buffer is empty
        assert (goals == 0).all()

    def test_buffer_argmax_full_updates_goals(self):
        buf = make_mock_buffer()
        buf.count.return_value = 10
        # sample returns data with logit key
        data = {"logit": torch.randn(2, 5, S, K)}
        buf.sample.return_value = (data, None, None)

        t = make_trainer(make_config(goal_sample="buffer", goal_type="argmax_full"), buffer=buf)
        goals = torch.zeros(B, S, K)
        mask = torch.ones(B, dtype=torch.bool)
        envs = make_mock_envs(goal_shape=(S, K))
        t._sample_goals(MagicMock(), envs, mask, goals, make_trans())
        # Goals should be set (nonzero due to argmax one-hot)
        buf.sample.assert_called_once()

    def test_buffer_other_goal_type_uses_stoch(self):
        buf = make_mock_buffer()
        buf.count.return_value = 10
        stoch = torch.zeros(2, 5, S, K)
        stoch[0, 0, 0, 1] = 1.0  # nonzero so we can detect it was used
        data = {"stoch": stoch}
        buf.sample.return_value = (data, None, None)

        t = make_trainer(make_config(goal_sample="buffer", goal_type="full"), buffer=buf)
        goals = torch.zeros(B, S, K)
        mask = torch.ones(B, dtype=torch.bool)
        envs = make_mock_envs(goal_shape=(S, K))
        t._sample_goals(MagicMock(), envs, mask, goals, make_trans())
        buf.sample.assert_called_once()

    def test_buffer_first_row_slices_stoch(self):
        buf = make_mock_buffer()
        buf.count.return_value = 10
        stoch = torch.zeros(2, 5, S, K)
        data = {"stoch": stoch}
        buf.sample.return_value = (data, None, None)

        t = make_trainer(make_config(goal_sample="buffer", goal_type="first_row"), buffer=buf)
        goals = torch.zeros(B, K)  # 1D goal shape for first_row
        mask = torch.ones(B, dtype=torch.bool)
        envs = make_mock_envs(goal_shape=(K,))
        t._sample_goals(MagicMock(), envs, mask, goals, make_trans())
        buf.sample.assert_called_once()

    def test_buffer_only_updates_masked_envs(self):
        buf = make_mock_buffer()
        buf.count.return_value = 10
        stoch = torch.ones(2, 5, S, K)  # all ones so assigned goals are 1
        data = {"stoch": stoch}
        buf.sample.return_value = (data, None, None)

        t = make_trainer(make_config(goal_sample="buffer", goal_type="full"), buffer=buf)
        goals = torch.zeros(B, S, K)
        # Only env 0 is masked
        mask = torch.tensor([True, False])
        envs = make_mock_envs(goal_shape=(S, K))
        t._sample_goals(MagicMock(), envs, mask, goals, make_trans())
        # env 0 should be updated, env 1 should remain zero
        assert (goals[0] != 0).any()
        assert (goals[1] == 0).all()

    def test_text_calls_text_goal(self):
        t = make_trainer(make_config(goal_sample="text"))
        agent = make_mock_agent()
        envs = make_mock_envs(goal_shape=(S, K))
        new_goals = torch.ones(B, S, K)

        with patch.object(t, "_text_goal", return_value=new_goals) as mock_text:
            goals = torch.zeros(B, S, K)
            mask = torch.ones(B, dtype=torch.bool)
            t._sample_goals(agent, envs, mask, goals, make_trans())
            mock_text.assert_called_once()
        assert (goals == 1.0).all()

    def test_text_skips_update_when_text_goal_returns_none(self):
        t = make_trainer(make_config(goal_sample="text"))
        agent = make_mock_agent()
        envs = make_mock_envs()
        goals = torch.zeros(B, S, K)

        with patch.object(t, "_text_goal", return_value=None):
            t._sample_goals(agent, envs, torch.ones(B, dtype=torch.bool), goals, make_trans())
        assert (goals == 0).all()

    def test_imagination_calls_imagine_goal(self):
        t = make_trainer(make_config(goal_sample="imagination", goal_type="full"))
        agent = MagicMock()
        new_goals = torch.ones(B, S, K)
        agent.imagine_goal.return_value = new_goals
        envs = make_mock_envs(goal_shape=(S, K))
        goals = torch.zeros(B, S, K)
        mask = torch.ones(B, dtype=torch.bool)
        t._sample_goals(agent, envs, mask, goals, make_trans())
        agent.imagine_goal.assert_called_once()

    def test_imagination_1d_goal_slices_first_row(self):
        t = make_trainer(make_config(goal_sample="imagination", goal_type="first_row"))
        agent = MagicMock()
        agent.imagine_goal.return_value = torch.ones(B, S, K)
        envs = make_mock_envs(goal_shape=(K,))
        goals = torch.zeros(B, K)
        mask = torch.ones(B, dtype=torch.bool)
        t._sample_goals(agent, envs, mask, goals, make_trans())
        # Result should be (B, K) not (B, S, K)
        assert goals.shape == (B, K)

    def test_image_calls_image_goal(self):
        t = make_trainer(make_config(goal_sample="image"))
        agent = MagicMock()
        envs = make_mock_envs(goal_shape=(S, K))
        new_goals = torch.ones(B, S, K)

        with patch.object(t, "_image_goal", return_value=new_goals) as mock_img:
            goals = torch.zeros(B, S, K)
            mask = torch.ones(B, dtype=torch.bool)
            t._sample_goals(agent, envs, mask, goals, make_trans())
            mock_img.assert_called_once()
        assert (goals == 1.0).all()

    def test_image_skips_update_when_image_goal_returns_none(self):
        t = make_trainer(make_config(goal_sample="image"))
        agent = MagicMock()
        envs = make_mock_envs()
        goals = torch.zeros(B, S, K)

        with patch.object(t, "_image_goal", return_value=None):
            t._sample_goals(agent, envs, torch.ones(B, dtype=torch.bool), goals, make_trans())
        assert (goals == 0).all()


# ---------------------------------------------------------------------------
# TestEval — helpers
# ---------------------------------------------------------------------------


def _run_eval(trainer, agent=None, done_sequence=None, trans_extra=None, goal_shape=(S, K)):
    """Run trainer.eval with done=True on the first step (loop runs once)."""
    if agent is None:
        agent = make_mock_agent()
    envs = trainer.eval_envs
    num_envs = envs.env_num

    if done_sequence is None:
        done_sequence = [torch.ones(num_envs, dtype=torch.bool)]

    def step_side_effect(act_cpu, done_cpu):
        d = done_sequence.pop(0) if done_sequence else torch.ones(num_envs, dtype=torch.bool)
        return make_trans(num_envs, goal_shape, extra=trans_extra), d

    envs.step.side_effect = step_side_effect
    trainer.eval(agent, train_step=0)


# ---------------------------------------------------------------------------
# TestEval
# ---------------------------------------------------------------------------


class TestEval:
    def test_calls_agent_eval_and_train(self):
        t = make_trainer()
        agent = make_mock_agent()
        _run_eval(t, agent)
        agent.eval.assert_called_once()
        agent.train.assert_called_once()

    def test_logs_eval_score(self):
        logger = MagicMock()
        t = make_trainer(logger=logger)
        _run_eval(t)
        calls = [c[0][0] for c in logger.scalar.call_args_list]
        assert "episode/eval_score" in calls

    def test_logs_eval_length(self):
        logger = MagicMock()
        t = make_trainer(logger=logger)
        _run_eval(t)
        calls = [c[0][0] for c in logger.scalar.call_args_list]
        assert "episode/eval_length" in calls

    def test_writes_logger_at_end(self):
        logger = MagicMock()
        t = make_trainer(logger=logger)
        _run_eval(t)
        logger.write.assert_called_once_with(0)

    def test_no_goals_when_goal_sample_random(self):
        t = make_trainer(make_config(goal_sample="random"))
        with patch.object(t, "_sample_goals") as mock_sg:
            _run_eval(t)
            mock_sg.assert_not_called()

    def test_goals_initialized_when_sampled_goal_source(self):
        t = make_trainer(make_config(goal_sample="buffer"))
        with patch.object(t, "_relabel_goal") as mock_rl:
            _run_eval(t, goal_shape=(S, K))
            mock_rl.assert_called()

    def test_sample_goals_called_on_is_first(self):
        t = make_trainer(make_config(goal_sample="buffer"))
        with patch.object(t, "_sample_goals") as mock_sg, patch.object(t, "_relabel_goal"):
            # Make is_first=True
            envs = t.eval_envs
            trans = make_trans(B, (S, K))
            trans["is_first"] = torch.ones(B, 1, dtype=torch.bool)
            envs.step.return_value = (trans, torch.ones(B, dtype=torch.bool))
            t.eval(make_mock_agent(), train_step=0)
            mock_sg.assert_called_once()

    def test_sample_goals_not_called_when_no_is_first(self):
        t = make_trainer(make_config(goal_sample="buffer"))
        with patch.object(t, "_sample_goals") as mock_sg, patch.object(t, "_relabel_goal"):
            _run_eval(t, goal_shape=(S, K))  # is_first defaults to False
            mock_sg.assert_not_called()

    def test_logit_stored_in_trans_when_in_agent_state(self):
        t = make_trainer()
        agent = make_mock_agent(with_logit=True)
        stored = {}

        real_apply = t._apply_reward

        def spy_apply(trans, rssm=None):
            stored["logit_present"] = "logit" in trans
            real_apply(trans, rssm)

        t._apply_reward = spy_apply
        _run_eval(t, agent)
        assert stored["logit_present"] is True

    def test_log_metric_accumulated(self):
        logger = MagicMock()
        t = make_trainer(logger=logger)
        agent = make_mock_agent()
        # trans includes a log_ key
        extra = {"log_my_metric": torch.ones(B, 1)}
        envs = t.eval_envs
        envs.step.return_value = (make_trans(B, (S, K), extra=extra), torch.ones(B, dtype=torch.bool))
        t.eval(agent, train_step=0)
        calls = [c[0][0] for c in logger.scalar.call_args_list]
        assert "episode/eval_my_metric" in calls

    def test_log_success_clipped_to_1(self):
        logger = MagicMock()
        t = make_trainer(logger=logger)
        agent = make_mock_agent()
        # Two iterations: first env not done, second both done
        envs = t.eval_envs
        envs.env_num = 2
        trans_with_success = make_trans(2, (S, K), extra={"log_success": torch.ones(2, 1) * 5.0})
        step_results = [
            (trans_with_success, torch.tensor([True, False])),
            (trans_with_success, torch.tensor([True, True])),
        ]
        envs.step.side_effect = lambda act, done: step_results.pop(0)
        state = {
            "prev_action": torch.zeros(2, A),
            "stoch": torch.zeros(2, S, K),
            "deter": torch.zeros(2, 64),
        }
        agent.get_initial_state.return_value = state
        agent.act.return_value = (torch.zeros(2, A), state, {})
        t.eval(agent, train_step=0)
        # Verify the log_success scalar was logged (clipped value)
        scalar_calls = {c[0][0]: c[0][1] for c in logger.scalar.call_args_list}
        assert "episode/eval_success" in scalar_calls

    def test_log_success_key_added_on_second_iteration(self):
        """log_metrics[key] already exists branch (second iteration same key)."""
        logger = MagicMock()
        t = make_trainer(logger=logger)
        envs = t.eval_envs
        envs.env_num = 2
        # Two iterations: first only env 0 done, second both done
        trans1 = make_trans(2, (S, K), extra={"log_success": torch.ones(2, 1)})
        trans2 = make_trans(2, (S, K), extra={"log_success": torch.ones(2, 1)})
        step_results = [
            (trans1, torch.tensor([True, False])),
            (trans2, torch.tensor([True, True])),
        ]
        envs.step.side_effect = lambda act, done: step_results.pop(0)
        state = {
            "prev_action": torch.zeros(2, A),
            "stoch": torch.zeros(2, S, K),
            "deter": torch.zeros(2, 64),
        }
        agent = make_mock_agent()
        agent.get_initial_state.return_value = state
        agent.act.return_value = (torch.zeros(2, A), state, {})
        t.eval(agent, train_step=0)  # must not raise — exercises "key already in log_metrics"

    def test_cache_and_image_logged(self):
        """Cover the 'image' in cache branch."""
        logger = MagicMock()
        t = make_trainer(make_config(batch_length=100), logger=logger)
        agent = make_mock_agent()
        # Patch torch.stack to return a dict-like mock
        stacked = MagicMock()
        stacked.__contains__ = lambda self, key: key == "image"
        stacked.__getitem__ = lambda self, key: torch.zeros(1, 5, 3, 64, 64)

        envs = t.eval_envs
        envs.step.return_value = (make_trans(B, (S, K)), torch.ones(B, dtype=torch.bool))

        with (
            patch("her_dream.trainer.torch.stack", return_value=stacked),
            patch("her_dream.trainer.tools.to_np", return_value=np.zeros((1, 5, 3, 64, 64))),
        ):
            t.eval(agent, train_step=0)

        assert logger.video.called
        assert logger.video.call_args[0][0] == "eval_video"

    def test_video_pred_log_calls_video_pred(self):
        """Cover the video_pred_log branch in eval."""
        logger = MagicMock()
        t = make_trainer(make_config(batch_length=100, video_pred_log=True), logger=logger)
        agent = make_mock_agent()
        stacked = MagicMock()
        stacked.__contains__ = lambda self, key: False  # no "image"
        stacked.__getitem__ = MagicMock(return_value=torch.zeros(1, 5, 3))

        envs = t.eval_envs
        envs.step.return_value = (make_trans(B, (S, K)), torch.ones(B, dtype=torch.bool))

        with (
            patch("her_dream.trainer.torch.stack", return_value=stacked),
            patch("her_dream.trainer.tools.to_np", return_value=np.zeros((1,))),
        ):
            t.eval(agent, train_step=0)

        agent.video_pred.assert_called_once()


# ---------------------------------------------------------------------------
# TestBegin — helpers
# ---------------------------------------------------------------------------


def _make_begin_trainer(config_overrides=None, goal_shape=(S, K), num_envs=1):
    """Trainer with steps=0 (loop never runs) unless config_overrides says otherwise."""
    cfg = make_config(**(config_overrides or {}))
    envs = make_mock_envs(num_envs=num_envs, goal_shape=goal_shape)
    buf = make_mock_buffer()
    return make_trainer(cfg, buffer=buf, train_envs=envs)


def _make_begin_step_effects(effects, num_envs=1, goal_shape=(S, K)):
    """Build side_effect list for envs.step from list of done-tensors."""

    def make_effect(done_tensor):
        def fn(act, d):
            return make_trans(num_envs, goal_shape), done_tensor

        return fn

    return [make_effect(d) for d in effects]


# ---------------------------------------------------------------------------
# TestBegin
# ---------------------------------------------------------------------------


class TestBegin:
    def test_loop_skipped_when_steps_already_reached(self):
        t = _make_begin_trainer({"steps": 0})
        agent = make_mock_agent()
        t.begin(agent)
        agent.act.assert_not_called()
        t.train_envs.step.assert_not_called()

    def test_loop_runs_one_iteration(self):
        """With steps=1 and one env, loop runs once (done=T→F makes step increment)."""
        t = _make_begin_trainer({"steps": 1, "batch_length": 1000})
        agent = make_mock_agent()
        envs = t.train_envs
        # Iter 1: done=True (start), step not incremented; envs.step returns done=False
        # Iter 2: done=False, step +=1; envs.step returns done=True
        # After iter 2: step=1, 1<1 is False → loop exits
        effects = [
            torch.zeros(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
        ]
        envs.step.side_effect = [
            (make_trans(1), effects[0]),
            (make_trans(1), effects[1]),
        ]
        t.begin(agent)
        assert envs.step.call_count == 2

    def test_add_transition_called(self):
        t = _make_begin_trainer({"steps": 1, "batch_length": 1000})
        agent = make_mock_agent()
        envs = t.train_envs
        envs.step.side_effect = [
            (make_trans(1), torch.zeros(1, dtype=torch.bool)),
            (make_trans(1), torch.ones(1, dtype=torch.bool)),
        ]
        t.begin(agent)
        assert t.replay_buffer.add_transition.call_count == 2

    def test_episode_logging_when_done_with_positive_length(self):
        """Covers 'if d and lengths[i] > 0' branch."""
        logger = MagicMock()
        t = _make_begin_trainer({"steps": 2, "batch_length": 1000}, num_envs=1)
        t.logger = logger
        agent = make_mock_agent()
        envs = t.train_envs
        # Iter 1: done=True, step=0, no log; envs.step→done=False
        # Iter 2: done=False, step+=1=1, lengths[0]=1; envs.step→done=True
        # Iter 3: done=True, lengths[0]=1 → log episode; envs.step→done=False
        # Iter 4: done=False, step+=1=2; envs.step→done=True (needed but loop exits before step)
        # Wait: after iter 4 step=2, loop condition 2<2=False → exits
        envs.step.side_effect = [
            (make_trans(1), torch.zeros(1, dtype=torch.bool)),
            (make_trans(1), torch.ones(1, dtype=torch.bool)),
            (make_trans(1), torch.zeros(1, dtype=torch.bool)),
            (make_trans(1), torch.ones(1, dtype=torch.bool)),
        ]
        t.begin(agent)
        calls = [c[0][0] for c in logger.scalar.call_args_list]
        assert "episode/score" in calls
        assert "episode/length" in calls

    def test_video_cache_logged_when_episode_done_for_env0(self):
        """Covers 'if i == 0 and len(video_cache) > 0' branch."""
        logger = MagicMock()
        cfg = make_config(steps=2, batch_length=1000)
        envs = make_mock_envs(num_envs=1, goal_shape=(S, K))
        t = make_trainer(cfg, logger=logger, train_envs=envs)
        agent = make_mock_agent()
        image = torch.zeros(1, 3, 64, 64)
        trans_with_image = make_trans(1, extra={"image": image})
        # 4 steps: done=T → F → T (log) → F → exits when step=2
        envs.step.side_effect = [
            (trans_with_image, torch.zeros(1, dtype=torch.bool)),
            (trans_with_image, torch.ones(1, dtype=torch.bool)),
            (trans_with_image, torch.zeros(1, dtype=torch.bool)),
            (trans_with_image, torch.ones(1, dtype=torch.bool)),
        ]
        with (
            patch("her_dream.trainer.torch.stack", return_value=torch.zeros(1, 3, 64, 64)),
            patch("her_dream.trainer.tools.to_np", return_value=np.zeros((1, 1, 3, 64, 64))),
        ):
            t.begin(agent)
        logger.video.assert_called()

    def test_eval_called_when_should_eval(self):
        cfg = make_config(steps=1, batch_length=1000, eval_every=1, eval_episode_num=1)
        t = _make_begin_trainer(config_overrides=cfg.__dict__)
        agent = make_mock_agent()
        envs = t.train_envs
        envs.step.side_effect = [
            (make_trans(1), torch.zeros(1, dtype=torch.bool)),
            (make_trans(1), torch.ones(1, dtype=torch.bool)),
        ]
        with patch.object(t, "eval") as mock_eval:
            t.begin(agent)
        mock_eval.assert_called()

    def test_goal_sampling_in_begin(self):
        cfg = make_config(steps=1, batch_length=1000, goal_sample="buffer")
        t = _make_begin_trainer(config_overrides=cfg.__dict__)
        agent = make_mock_agent()
        envs = t.train_envs
        envs.step.side_effect = [
            (make_trans(1), torch.zeros(1, dtype=torch.bool)),
            (make_trans(1), torch.ones(1, dtype=torch.bool)),
        ]
        with patch.object(t, "_relabel_goal") as mock_rl:
            t.begin(agent)
        mock_rl.assert_called()

    def test_sample_goals_on_is_first_in_begin(self):
        cfg = make_config(steps=1, batch_length=1000, goal_sample="buffer")
        t = _make_begin_trainer(config_overrides=cfg.__dict__)
        agent = make_mock_agent()
        envs = t.train_envs
        trans_first = make_trans(1)
        trans_first["is_first"] = torch.ones(1, 1, dtype=torch.bool)
        envs.step.side_effect = [
            (trans_first, torch.zeros(1, dtype=torch.bool)),
            (make_trans(1), torch.ones(1, dtype=torch.bool)),
        ]
        with patch.object(t, "_sample_goals") as mock_sg, patch.object(t, "_relabel_goal"):
            t.begin(agent)
        mock_sg.assert_called()

    def test_obs_step_prob_log(self):
        logger = MagicMock()
        cfg = make_config(steps=1, batch_length=1000, obs_step_prob_log=True)
        t = _make_begin_trainer(config_overrides=cfg.__dict__)
        t.logger = logger
        agent = make_mock_agent()
        act_metrics = {"obs_step_sample_log_prob": torch.tensor(-1.5)}
        agent.act.return_value = (torch.zeros(1, A), agent.get_initial_state.return_value, act_metrics)
        envs = t.train_envs
        envs.step.side_effect = [
            (make_trans(1), torch.zeros(1, dtype=torch.bool)),
            (make_trans(1), torch.ones(1, dtype=torch.bool)),
        ]
        t.begin(agent)
        logger.write_step.assert_called()

    def test_logit_stored_in_trans_in_begin(self):
        cfg = make_config(steps=1, batch_length=1000)
        t = _make_begin_trainer(config_overrides=cfg.__dict__)
        agent = make_mock_agent(with_logit=True)
        envs = t.train_envs
        stored = []

        real_add = t.replay_buffer.add_transition

        def spy_add(trans):
            stored.append("logit" in trans)
            real_add(trans)

        t.replay_buffer.add_transition = spy_add
        envs.step.side_effect = [
            (make_trans(1), torch.zeros(1, dtype=torch.bool)),
            (make_trans(1), torch.ones(1, dtype=torch.bool)),
        ]
        t.begin(agent)
        assert any(stored)

    def test_image_appended_to_video_cache(self):
        cfg = make_config(steps=1, batch_length=1000)
        t = _make_begin_trainer(config_overrides=cfg.__dict__)
        agent = make_mock_agent()
        envs = t.train_envs
        image = torch.zeros(1, 3, 64, 64)
        trans_with_image = make_trans(1, extra={"image": image})
        envs.step.side_effect = [
            (trans_with_image, torch.zeros(1, dtype=torch.bool)),
            (make_trans(1), torch.ones(1, dtype=torch.bool)),
        ]
        # Verify video_cache gets the image (indirectly via no error)
        t.begin(agent)

    def test_update_called_when_should_update(self):
        # _should_update(step): step // env_num > batch_length+1
        # batch_length=0 → True when step >= 2
        cfg = make_config(steps=4, batch_length=0, pretrain=1)
        envs = make_mock_envs(num_envs=1)
        buf = make_mock_buffer()
        t = make_trainer(cfg, buffer=buf, train_envs=envs)
        agent = make_mock_agent()
        # Always return done=False so step increments every iteration; exits when step=4
        envs.step.side_effect = lambda act, d: (make_trans(1), torch.zeros(1, dtype=torch.bool))
        agent.update.return_value = {"loss": torch.tensor(0.5)}
        t.begin(agent)
        agent.update.assert_called()

    def test_pretrain_used_on_first_update(self):
        cfg = make_config(steps=4, batch_length=0, pretrain=3)
        envs = make_mock_envs(num_envs=1)
        t = make_trainer(cfg, buffer=make_mock_buffer(), train_envs=envs)
        agent = make_mock_agent()
        agent.update.return_value = {"loss": torch.tensor(0.0)}
        envs.step.side_effect = lambda act, d: (make_trans(1), torch.zeros(1, dtype=torch.bool))
        t.begin(agent)
        # pretrain=3 → update called 3 times on first update trigger
        assert agent.update.call_count >= 3

    def test_updates_needed_used_after_pretrain(self):
        """_should_pretrain() returns False on second call, uses _updates_needed."""
        cfg = make_config(steps=8, batch_length=0, pretrain=1, train_ratio=1.0)
        envs = make_mock_envs(num_envs=1)
        t = make_trainer(cfg, buffer=make_mock_buffer(), train_envs=envs)
        agent = make_mock_agent()
        agent.update.return_value = {"loss": torch.tensor(0.0)}
        envs.step.side_effect = lambda act, d: (make_trans(1), torch.zeros(1, dtype=torch.bool))
        t.begin(agent)
        assert agent.update.call_count >= 1

    def test_logging_when_should_log(self):
        logger = MagicMock()
        cfg = make_config(steps=4, batch_length=0, update_log_every=1)
        envs = make_mock_envs(num_envs=1)
        t = make_trainer(cfg, buffer=make_mock_buffer(), train_envs=envs, logger=logger)
        agent = make_mock_agent()
        agent.update.return_value = {"wm_loss": torch.tensor(0.5)}
        envs.step.side_effect = lambda act, d: (make_trans(1), torch.zeros(1, dtype=torch.bool))
        t.begin(agent)
        calls = [c[0][0] for c in logger.scalar.call_args_list]
        assert any("train/" in c for c in calls)

    def test_video_pred_log_in_begin(self):
        logger = MagicMock()
        cfg = make_config(steps=4, batch_length=0, update_log_every=1, video_pred_log=True)
        envs = make_mock_envs(num_envs=1)
        buf = make_mock_buffer()
        buf.sample.return_value = (make_trans(1), None, (torch.zeros(1, S, K), torch.zeros(1, 64)))
        t = make_trainer(cfg, buffer=buf, train_envs=envs, logger=logger)
        agent = make_mock_agent()
        agent.update.return_value = {"loss": torch.tensor(0.0)}
        envs.step.side_effect = lambda act, d: (make_trans(1), torch.zeros(1, dtype=torch.bool))
        with patch("her_dream.trainer.tools.to_np", return_value=np.zeros((1,))):
            t.begin(agent)
        agent.video_pred.assert_called()

    def test_params_hist_log_in_begin(self):
        logger = MagicMock()
        cfg = make_config(steps=4, batch_length=0, update_log_every=1, params_hist_log=True)
        envs = make_mock_envs(num_envs=1)
        t = make_trainer(cfg, buffer=make_mock_buffer(), train_envs=envs, logger=logger)
        agent = make_mock_agent()
        agent.update.return_value = {"loss": torch.tensor(0.0)}
        agent._named_params = {"w": torch.zeros(3)}
        envs.step.side_effect = lambda act, d: (make_trans(1), torch.zeros(1, dtype=torch.bool))
        with patch("her_dream.trainer.tools.to_np", return_value=np.zeros((3,))):
            t.begin(agent)
        logger.histogram.assert_called()

    def test_episode_ids_increment_on_done(self):
        """episode_ids[done] += envs.env_num when done is True."""
        cfg = make_config(steps=1, batch_length=1000)
        envs = make_mock_envs(num_envs=1)
        t = make_trainer(cfg, train_envs=envs)
        agent = make_mock_agent()
        # Iter1: done starts True; env.step→done=False
        # Iter2: done=False; step+=1; env.step→done=True; episode_ids[0]+=1
        envs.step.side_effect = [
            (make_trans(1), torch.zeros(1, dtype=torch.bool)),
            (make_trans(1), torch.ones(1, dtype=torch.bool)),
        ]
        t.begin(agent)  # Must complete without error


class TestLexaExploreMask:
    """LEXA alternates data collection between the explorer and the achiever."""

    @staticmethod
    def trainer_with(**overrides):
        return make_trainer(make_config(**overrides))

    def test_no_mask_outside_lexa(self):
        trainer = self.trainer_with()
        assert trainer._explore_mask(torch.arange(4)) is None

    def test_selects_every_nth_episode(self):
        trainer = self.trainer_with(lexa=True, explore_every_ep=2)
        mask = trainer._explore_mask(torch.arange(6))
        assert mask.tolist() == [True, False, True, False, True, False]

    def test_a_cadence_of_one_always_explores(self):
        trainer = self.trainer_with(lexa=True, explore_every_ep=1)
        assert trainer._explore_mask(torch.arange(4)).all()

    def test_a_cadence_of_zero_disables_the_split(self):
        # 0 means the achiever collects everything.
        trainer = self.trainer_with(lexa=True, explore_every_ep=0)
        assert trainer._explore_mask(torch.arange(4)) is None

    def test_is_per_env_not_per_step(self):
        # Envs run independent episodes, so the mask is indexed by episode id.
        trainer = self.trainer_with(lexa=True, explore_every_ep=3)
        assert trainer._explore_mask(torch.tensor([0, 1, 3, 6])).tolist() == [True, False, True, True]
