from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from .conftest import A, D, K, S, make_her_buffer, make_mock_buffer_for_goal, make_transition


class TestHERStrategyEnum:
    def test_enum_members_exist(self):
        from buffers.her_buffer import HERStrategy

        assert HERStrategy.FUTURE.value == 0
        assert HERStrategy.FINAL.value == 1
        assert HERStrategy.EPISODE.value == 2

    def test_enum_from_string(self):
        from buffers.her_buffer import HERStrategy

        assert HERStrategy["FUTURE"] == HERStrategy.FUTURE
        assert HERStrategy["FINAL"] == HERStrategy.FINAL
        assert HERStrategy["EPISODE"] == HERStrategy.EPISODE


class TestHERBufferInit:
    def test_ep_start_shape(self, her_buffer):
        # n_envs=2, max_columns=max_size/n_envs=20
        assert her_buffer.ep_start.shape == (2, 20)

    def test_ep_length_shape(self, her_buffer):
        assert her_buffer.ep_length.shape == (2, 20)

    def test_ep_start_initialized_to_zero(self, her_buffer):
        assert (her_buffer.ep_start == 0).all()

    def test_ep_length_initialized_to_zero(self, her_buffer):
        assert (her_buffer.ep_length == 0).all()

    def test_initial_pos_is_zero(self, her_buffer):
        assert her_buffer.pos == 0

    def test_current_ep_start_initialized_to_zero(self, her_buffer):
        assert (her_buffer._current_ep_start == 0).all()

    def test_her_ratio_set(self, her_buffer):
        assert her_buffer.her_ratio == 0.5


class TestComputeEpisodeLength:
    def test_length_accumulates_across_steps(self, her_buffer):
        env_id = 0

        # Simulate advancing pos manually and computing lengths step-by-step
        for expected_len in range(1, 4):
            her_buffer.pos = expected_len
            her_buffer._compute_episode_length(env_id, is_last=False)
            # All positions from 0 to pos-1 should have this episode length
            for idx in range(expected_len):
                assert her_buffer.ep_length[env_id, idx] == expected_len

    def test_episode_start_resets_after_is_last(self, her_buffer):
        env_id = 0
        her_buffer.pos = 3
        her_buffer._compute_episode_length(env_id, is_last=True)
        assert her_buffer._current_ep_start[env_id] == 3

    def test_episode_start_does_not_reset_mid_episode(self, her_buffer):
        env_id = 0
        her_buffer.pos = 3
        her_buffer._compute_episode_length(env_id, is_last=False)
        assert her_buffer._current_ep_start[env_id] == 0

    def test_wraparound_handles_circular_indices(self):
        buf = make_her_buffer(n_envs=2, max_size=20)  # max_columns = 10
        env_id = 0
        # Simulate episode starting near the end of the circular buffer
        buf._current_ep_start[env_id] = 8
        buf.pos = 2  # wrapped around: episode spans [8, 9, 0, 1]
        buf._compute_episode_length(env_id, is_last=False)
        for idx in [8, 9, 0, 1]:
            assert buf.ep_length[env_id, idx] == 4


class TestParseIndices:
    def test_reshapes_to_batch_and_sequence(self, her_buffer):
        B, T = 2, her_buffer.batch_length
        n = B * (T + 1)
        info = {"index": [torch.arange(n), torch.arange(n) % 2]}
        envs, steps = her_buffer._parse_indices(info)
        assert envs.shape == (B, T + 1)
        assert steps.shape == (B, T + 1)

    def test_values_preserved_after_reshape(self, her_buffer):
        B, T = 2, her_buffer.batch_length
        step_flat = torch.arange(B * (T + 1))
        info = {"index": [step_flat, torch.zeros(B * (T + 1), dtype=torch.long)]}
        _, steps = her_buffer._parse_indices(info)
        assert torch.equal(steps.reshape(-1), step_flat)


class TestGetValidBatches:
    def test_all_valid_when_all_lengths_positive(self, her_buffer):
        her_buffer.ep_length[:] = 5
        B, T = 2, her_buffer.batch_length
        envs = torch.zeros(B, T + 1, dtype=torch.long)
        steps = torch.zeros(B, T + 1, dtype=torch.long)
        result = her_buffer._get_valid_batches(envs, steps)
        assert result.all()

    def test_batch_invalid_when_any_step_has_zero_length(self, her_buffer):
        her_buffer.ep_length[:] = 5
        her_buffer.ep_length[0, 0] = 0  # invalidate one position
        B, T = 2, her_buffer.batch_length
        # batch 0 accesses (env=0, step=0) which has length 0
        envs = torch.zeros(B, T + 1, dtype=torch.long)
        steps = torch.zeros(B, T + 1, dtype=torch.long)
        result = her_buffer._get_valid_batches(envs, steps)
        assert not result[0]

    def test_valid_batch_unaffected_by_other_invalid(self, her_buffer):
        her_buffer.ep_length[:] = 5
        her_buffer.ep_length[0, 0] = 0
        B, T = 2, her_buffer.batch_length
        # batch 1 accesses env=1, step=0 which is still valid (length=5)
        envs = torch.tensor([[0] * (T + 1), [1] * (T + 1)])
        steps = torch.zeros(B, T + 1, dtype=torch.long)
        result = her_buffer._get_valid_batches(envs, steps)
        assert result[1]


class TestSampleGoal:
    T = 5  # sequence length used in goal tests

    def _setup_episode(self, buf, ep_start_val=0, ep_length_val=10):
        buf.ep_start[:] = ep_start_val
        buf.ep_length[:] = ep_length_val

    def test_final_strategy_returns_correct_shape(self):
        buf = make_her_buffer(goal_type="full", her_strategy="final")
        self._setup_episode(buf)
        buf._buffer = make_mock_buffer_for_goal(self.T)
        env_indices = np.zeros(self.T, dtype=np.int64)
        step_indices = np.zeros(self.T, dtype=np.int64)
        goal = buf._sample_goal(env_indices, step_indices)
        assert goal.shape == (self.T, S, K)

    def test_final_strategy_fetches_last_step(self):
        buf = make_her_buffer(goal_type="full", her_strategy="final")
        ep_len = 6
        self._setup_episode(buf, ep_start_val=0, ep_length_val=ep_len)
        mock_buf = MagicMock()
        captured = {}

        def capture_getitem(key):
            captured["key"] = key
            env_idx, t_idx = key
            return TensorDict({"stoch": torch.ones(len(env_idx), S, K)}, batch_size=[len(env_idx)])

        mock_buf.__getitem__.side_effect = capture_getitem
        buf._buffer = mock_buf
        env_indices = np.zeros(self.T, dtype=np.int64)
        step_indices = np.zeros(self.T, dtype=np.int64)
        buf._sample_goal(env_indices, step_indices)
        # FINAL: transition_indices_in_episode = ep_length - 1 = 5
        expected_t = (0 + ep_len - 1) % buf.max_columns
        assert (captured["key"][1] == expected_t).all()

    def test_future_strategy_returns_correct_shape(self):
        buf = make_her_buffer(goal_type="full", her_strategy="future")
        self._setup_episode(buf, ep_start_val=0, ep_length_val=10)
        buf._buffer = make_mock_buffer_for_goal(self.T)
        env_indices = np.zeros(self.T, dtype=np.int64)
        step_indices = np.ones(self.T, dtype=np.int64) * 3  # current step = 3
        goal = buf._sample_goal(env_indices, step_indices)
        assert goal.shape == (self.T, S, K)

    def test_episode_strategy_returns_correct_shape(self):
        buf = make_her_buffer(goal_type="full", her_strategy="episode")
        self._setup_episode(buf, ep_start_val=0, ep_length_val=8)
        buf._buffer = make_mock_buffer_for_goal(self.T)
        env_indices = np.zeros(self.T, dtype=np.int64)
        step_indices = np.zeros(self.T, dtype=np.int64)
        goal = buf._sample_goal(env_indices, step_indices)
        assert goal.shape == (self.T, S, K)

    def test_goal_type_first_row_slices_group_dim(self):
        buf = make_her_buffer(goal_type="first_row", her_strategy="final")
        self._setup_episode(buf)
        buf._buffer = make_mock_buffer_for_goal(self.T)
        env_indices = np.zeros(self.T, dtype=np.int64)
        step_indices = np.zeros(self.T, dtype=np.int64)
        goal = buf._sample_goal(env_indices, step_indices)
        # first_row: stoch[:, 0] → (T, K)
        assert goal.shape == (self.T, K)

    def test_invalid_strategy_raises_value_error(self, her_buffer):
        her_buffer.her_strategy = "invalid"
        env_indices = np.zeros(self.T, dtype=np.int64)
        step_indices = np.zeros(self.T, dtype=np.int64)
        with pytest.raises(ValueError):
            her_buffer._sample_goal(env_indices, step_indices)

    def test_argmax_full_goal_type_returns_one_hot(self):
        buf = make_her_buffer(goal_type="argmax_full", her_strategy="final")
        self._setup_episode(buf)
        buf._buffer = make_mock_buffer_for_goal(self.T)
        env_indices = np.zeros(self.T, dtype=np.int64)
        step_indices = np.zeros(self.T, dtype=np.int64)
        goal = buf._sample_goal(env_indices, step_indices)
        assert goal.dtype == torch.float32
        assert goal.sum(dim=-1).eq(1.0).all()

    def test_log_prob_goal_type_returns_one_hot(self):
        buf = make_her_buffer(goal_type="log_prob", her_strategy="final")
        self._setup_episode(buf)
        buf._buffer = make_mock_buffer_for_goal(self.T)
        env_indices = np.zeros(self.T, dtype=np.int64)
        step_indices = np.zeros(self.T, dtype=np.int64)
        goal = buf._sample_goal(env_indices, step_indices)
        assert goal.dtype == torch.float32
        assert goal.sum(dim=-1).eq(1.0).all()


class TestHERBufferAddTransition:
    def test_pos_advances_after_add(self, her_buffer):
        her_buffer.add_transition(make_transition())
        assert her_buffer.pos == 1

    def test_pos_wraps_at_max_columns(self):
        buf = make_her_buffer(n_envs=2, max_size=4)  # max_columns=2
        buf.add_transition(make_transition())
        buf.add_transition(make_transition())
        assert buf.pos == 0

    def test_ep_start_recorded_at_current_pos(self, her_buffer):
        her_buffer._current_ep_start[:] = 3
        her_buffer.add_transition(make_transition())
        assert (her_buffer.ep_start[:, 0] == 3).all()

    def test_base_buffer_receives_data(self, her_buffer):
        assert her_buffer.count() == 0
        her_buffer.add_transition(make_transition())
        assert her_buffer.count() > 0

    def test_ep_length_updated_after_add(self, her_buffer):
        her_buffer.add_transition(make_transition())
        assert (her_buffer.ep_length[:, 0] == 1).all()

    def test_is_last_resets_episode_start(self, her_buffer):
        data = make_transition()
        data.set_("is_last", torch.ones(2, dtype=torch.bool))
        her_buffer.add_transition(data)
        assert (her_buffer._current_ep_start == 1).all()

    def test_overwrite_zeros_old_episode_length(self):
        buf = make_her_buffer(n_envs=2, max_size=4)  # max_columns=2
        buf.add_transition(make_transition())  # pos: 0→1, ep_length[:,0]=1
        buf.add_transition(make_transition())  # pos: 1→0, ep_length[:,[0,1]]=2
        buf.add_transition(make_transition())  # pos: 0→1, invalidates [0,1], ep_length[:,0]=1
        assert (buf.ep_length[:, 1] == 0).all()


class TestHERBufferSample:
    def _capture_her_indices(self, buf):
        captured = {}

        def passthrough(sample_td, envs, steps, her_indices):
            captured["indices"] = her_indices
            return sample_td

        with patch.object(buf, "_apply_her", side_effect=passthrough):
            buf.sample()
        return captured["indices"]

    def test_sample_output_shapes(self):
        buf = make_her_buffer(her_ratio=0.0)
        for _ in range(6):
            buf.add_transition(make_transition())
        data, index, (init_stoch, init_deter) = buf.sample()
        B, T = buf.batch_size, buf.batch_length
        assert data["stoch"].shape == (B, T, S, K)
        assert data["deter"].shape == (B, T, D)
        assert data["action"].shape == (B, T, A)
        assert index[0].shape == (B, T)
        assert index[1].shape == (B, T)
        assert init_stoch.shape == (B, S, K)
        assert init_deter.shape == (B, D)

    def test_apply_her_called_with_non_empty_indices(self, filled_her_buffer):
        indices = self._capture_her_indices(filled_her_buffer)
        assert len(indices) > 0

    def test_n_her_bounded_by_her_ratio(self, filled_her_buffer):
        indices = self._capture_her_indices(filled_her_buffer)
        assert len(indices) <= filled_her_buffer.her_ratio * filled_her_buffer.batch_size

    def test_no_her_when_her_ratio_zero(self):
        buf = make_her_buffer(her_ratio=0.0)
        for _ in range(6):
            buf.add_transition(make_transition())
        indices = self._capture_her_indices(buf)
        assert len(indices) == 0


class TestApplyHER:
    B = 2
    T1 = 5  # batch_length + 1

    def _make_sample_td(self):
        return TensorDict(
            {
                "stoch": torch.zeros(self.B, self.T1, S, K),
                "logit": torch.zeros(self.B, self.T1, S, K),
                "goal": torch.zeros(self.B, self.T1, K),
                "reward": torch.zeros(self.B, self.T1),
            },
            batch_size=[self.B, self.T1],
        )

    def _make_envs_steps(self):
        return (
            torch.zeros(self.B, self.T1, dtype=torch.long),
            torch.zeros(self.B, self.T1, dtype=torch.long),
        )

    def test_empty_indices_returns_td_unchanged(self, her_buffer):
        sample_td = self._make_sample_td()
        envs, steps = self._make_envs_steps()
        original_goal = sample_td["goal"].clone()
        result = her_buffer._apply_her(sample_td, envs, steps, np.array([]))
        assert torch.equal(result["goal"], original_goal)

    def test_goal_updated_for_her_batch(self, her_buffer):
        sample_td = self._make_sample_td()
        envs, steps = self._make_envs_steps()
        sentinel = torch.ones(self.T1, K)
        with patch.object(her_buffer, "_sample_goal", return_value=sentinel):
            result = her_buffer._apply_her(sample_td, envs, steps, np.array([0]))
        assert torch.equal(result["goal"][0], sentinel)

    def test_non_her_batch_goal_unchanged(self, her_buffer):
        sample_td = self._make_sample_td()
        envs, steps = self._make_envs_steps()
        original_goal_1 = sample_td["goal"][1].clone()
        sentinel = torch.ones(self.T1, K)
        with patch.object(her_buffer, "_sample_goal", return_value=sentinel):
            result = her_buffer._apply_her(sample_td, envs, steps, np.array([0]))
        assert torch.equal(result["goal"][1], original_goal_1)

    def test_reward_recomputed_for_her_batch(self, her_buffer):
        sample_td = self._make_sample_td()
        envs, steps = self._make_envs_steps()
        sentinel_goal = torch.ones(self.T1, K)
        sentinel_reward = 99.0
        her_buffer.reward_function = lambda a, d: torch.full((a.shape[0],), sentinel_reward)
        with patch.object(her_buffer, "_sample_goal", return_value=sentinel_goal):
            result = her_buffer._apply_her(sample_td, envs, steps, np.array([0]))
        assert result["reward"][0].eq(sentinel_reward).all()

    def test_log_prob_goal_type_calls_rssm_get_dist(self):
        buf = make_her_buffer(goal_type="log_prob")
        mock_rssm = MagicMock()
        buf.rssm = mock_rssm
        buf.reward_function = lambda a, d: torch.zeros(self.T1)
        sample_td = self._make_sample_td()
        envs, steps = self._make_envs_steps()
        sentinel_goal = torch.ones(self.T1, K)
        with patch.object(buf, "_sample_goal", return_value=sentinel_goal):
            buf._apply_her(sample_td, envs, steps, np.array([0]))
        mock_rssm.get_dist.assert_called_once()
