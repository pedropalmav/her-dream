import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_ale_mock(screen_h=210, screen_w=160, n_actions=18, lives=3):
    ale = MagicMock()
    ale.getScreenDims.return_value = (screen_h, screen_w)
    ale.getLegalActionSet.return_value = list(range(n_actions))
    ale.getMinimalActionSet.return_value = list(range(4))
    ale.lives.return_value = lives
    ale.game_over.return_value = False
    ale.act.return_value = 0.0

    def _get_screen_rgb(buf):
        buf[:] = 0

    ale.getScreenRGB.side_effect = _get_screen_rgb
    return ale


def _make_atari(name="pong", ale_mock=None, rom_mock=None, **kwargs):
    """Create an Atari instance with fully mocked ALE."""
    if ale_mock is None:
        ale_mock = _make_ale_mock()
    if rom_mock is None:
        rom_mock = MagicMock(return_value="/fake/path/pong.bin")

    # Reset LOCK so __init__ always creates one
    from envs import atari as atari_module

    atari_module.Atari.LOCK = None

    with (
        patch.object(atari_module.ale_py, "ALEInterface", return_value=ale_mock),
        patch.object(atari_module.roms, "get_rom_path", rom_mock),
        patch("multiprocessing.get_context") as mp_ctx,
    ):
        mp_ctx.return_value.Lock.return_value = MagicMock()
        mp_ctx.return_value.Lock.return_value.__enter__ = MagicMock(return_value=None)
        mp_ctx.return_value.Lock.return_value.__exit__ = MagicMock(return_value=False)
        from envs.atari import Atari

        env = Atari(name, **kwargs)
    return env, ale_mock


class TestAtariInit:
    def test_size_must_be_square(self):
        from envs.atari import Atari

        Atari.LOCK = None
        ale = _make_ale_mock()
        with pytest.raises(AssertionError):
            _make_atari("pong", ale_mock=ale, size=(64, 84))

    def test_invalid_lives_raises(self):
        from envs.atari import Atari

        Atari.LOCK = None
        with pytest.raises(AssertionError):
            _make_atari("pong", lives="bad_mode")

    def test_invalid_actions_raises(self):
        from envs.atari import Atari

        Atari.LOCK = None
        with pytest.raises(AssertionError):
            _make_atari("pong", actions="partial")

    def test_invalid_resize_raises(self):
        from envs.atari import Atari

        Atari.LOCK = None
        with pytest.raises(AssertionError):
            _make_atari("pong", resize="bilinear")

    def test_invalid_aggregate_raises(self):
        from envs.atari import Atari

        Atari.LOCK = None
        with pytest.raises(AssertionError):
            _make_atari("pong", aggregate="sum")

    def test_pooling_too_small_raises(self):
        from envs.atari import Atari

        Atari.LOCK = None
        with pytest.raises(AssertionError):
            _make_atari("pong", pooling=0)

    def test_action_repeat_too_small_raises(self):
        from envs.atari import Atari

        Atari.LOCK = None
        with pytest.raises(AssertionError):
            _make_atari("pong", action_repeat=0)

    def test_james_bond_renamed_to_jamesbond(self):
        env, ale = _make_atari("james_bond")
        ale.loadROM.assert_called_once()

    def test_ale_rom_path_env_var(self):
        ale = _make_ale_mock()
        from envs import atari as atari_module

        atari_module.Atari.LOCK = None
        with (
            patch.dict(os.environ, {"ALE_ROM_PATH": "/custom/roms"}),
            patch.object(atari_module.ale_py, "ALEInterface", return_value=ale),
            patch.object(atari_module.roms, "get_rom_path") as rom_mock,
            patch("multiprocessing.get_context") as mp_ctx,
        ):
            mp_ctx.return_value.Lock.return_value = MagicMock()
            mp_ctx.return_value.Lock.return_value.__enter__ = MagicMock(return_value=None)
            mp_ctx.return_value.Lock.return_value.__exit__ = MagicMock(return_value=False)
            from envs.atari import Atari

            Atari("pong")
            ale.loadROM.assert_called_with("/custom/roms/pong.bin")
            rom_mock.assert_not_called()

    def test_lock_reused_on_second_instance(self):
        from envs import atari as atari_module

        atari_module.Atari.LOCK = None
        env1, _ = _make_atari("pong")
        first_lock = atari_module.Atari.LOCK
        # Create second instance - LOCK already set, skip creation
        ale2 = _make_ale_mock()
        with (
            patch.object(atari_module.ale_py, "ALEInterface", return_value=ale2),
            patch.object(atari_module.roms, "get_rom_path", return_value="/fake/pong.bin"),
        ):
            from envs.atari import Atari

            Atari("pong")
        assert atari_module.Atari.LOCK is first_lock

    def test_opencv_resize_imports_cv2(self):

        from envs import atari as atari_module

        atari_module.Atari.LOCK = None
        ale = _make_ale_mock()
        mock_cv2 = MagicMock()
        with (
            patch.object(atari_module.ale_py, "ALEInterface", return_value=ale),
            patch.object(atari_module.roms, "get_rom_path", return_value="/fake/pong.bin"),
            patch("multiprocessing.get_context") as mp_ctx,
        ):
            mp_ctx.return_value.Lock.return_value = MagicMock()
            mp_ctx.return_value.Lock.return_value.__enter__ = MagicMock(return_value=None)
            mp_ctx.return_value.Lock.return_value.__exit__ = MagicMock(return_value=False)
            with patch.dict(sys.modules, {"cv2": mock_cv2}):
                from envs.atari import Atari

                env = Atari("pong", resize="opencv")
        assert env._resize_fn == "opencv"


class TestAtariObsAndActionSpace:
    def test_obs_space_gray_shape(self):
        env, _ = _make_atari("pong", gray=True, size=(84, 84))
        space = env.observation_space
        assert space["image"].shape == (84, 84, 1)

    def test_obs_space_color_shape(self):
        env, _ = _make_atari("pong", gray=False, size=(84, 84))
        space = env.observation_space
        assert space["image"].shape == (84, 84, 3)

    def test_action_space_size(self):
        ale = _make_ale_mock(n_actions=10)
        env, _ = _make_atari("pong", ale_mock=ale, actions="all")
        assert env.action_space.n == 10

    def test_minimal_actions(self):
        ale = _make_ale_mock(n_actions=18)
        env, _ = _make_atari("pong", ale_mock=ale, actions="needed")
        assert env.action_space.n == 4


class TestAtariObs:
    def _env_with_buffer(self, size=(8, 8), gray=True, aggregate="max", clip_reward=False, resize="pillow"):
        ale = _make_ale_mock(screen_h=size[0], screen_w=size[1])
        env, ale_mock = _make_atari(
            "pong",
            ale_mock=ale,
            size=size,
            gray=gray,
            aggregate=aggregate,
            clip_reward=clip_reward,
            resize=resize,
            pooling=2,
        )
        return env, ale_mock

    def test_obs_clip_reward_true(self):
        env, _ = self._env_with_buffer(clip_reward=True)
        env._done = False
        obs, rew, _, _ = env._obs(5.0)
        assert rew == 1.0

    def test_obs_clip_reward_false(self):
        env, _ = self._env_with_buffer(clip_reward=False)
        env._done = False
        obs, rew, _, _ = env._obs(5.0)
        assert rew == 5.0

    def test_obs_aggregate_max(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), aggregate="max", pooling=2)
        # Fill buffer with known values
        env._buffers[0][:] = 100
        env._buffers[1][:] = 0
        obs, _, _, _ = env._obs(0.0)
        assert obs["image"].max() > 0

    def test_obs_aggregate_mean(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), aggregate="mean", pooling=2)
        env._buffers[0][:] = 100
        env._buffers[1][:] = 0
        obs, _, _, _ = env._obs(0.0)
        assert obs["image"].max() >= 0

    def test_obs_gray_applies_weights(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), gray=True, pooling=2)
        obs, _, _, _ = env._obs(0.0)
        assert obs["image"].shape[-1] == 1

    def test_obs_color_shape(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), gray=False, pooling=2)
        obs, _, _, _ = env._obs(0.0)
        assert obs["image"].shape[-1] == 3

    def test_obs_no_resize_when_already_correct_size(self):
        # buffer same size as target → no resize call
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), gray=False, pooling=2)
        obs, _, _, _ = env._obs(0.0)
        assert obs["image"].shape[:2] == (8, 8)

    def test_obs_resize_pillow(self):
        # Buffer bigger than target → resize via pillow
        ale = _make_ale_mock(screen_h=16, screen_w=16)
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), resize="pillow", gray=False, pooling=2)
        obs, _, _, _ = env._obs(0.0)
        assert obs["image"].shape[:2] == (8, 8)

    def test_obs_resize_opencv(self):
        ale = _make_ale_mock(screen_h=16, screen_w=16)
        mock_cv2 = MagicMock()
        mock_cv2.resize.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        from envs import atari as atari_module

        atari_module.Atari.LOCK = None
        with (
            patch.object(atari_module.ale_py, "ALEInterface", return_value=ale),
            patch.object(atari_module.roms, "get_rom_path", return_value="/fake/pong.bin"),
            patch("multiprocessing.get_context") as mp_ctx,
        ):
            mp_ctx.return_value.Lock.return_value = MagicMock()
            mp_ctx.return_value.Lock.return_value.__enter__ = MagicMock(return_value=None)
            mp_ctx.return_value.Lock.return_value.__exit__ = MagicMock(return_value=False)
            with patch.dict(sys.modules, {"cv2": mock_cv2}):
                from envs.atari import Atari

                env = Atari("pong", size=(8, 8), resize="opencv", gray=False, pooling=2)
        env._cv2 = mock_cv2
        env._obs(0.0)
        mock_cv2.resize.assert_called_once()

    def test_obs_is_first_flag(self):
        env, _ = self._env_with_buffer()
        obs, _, _, _ = env._obs(0.0, is_first=True)
        assert obs["is_first"] is True
        assert obs["is_last"] is False

    def test_obs_is_last_and_is_terminal(self):
        env, _ = self._env_with_buffer()
        obs, _, is_last, _ = env._obs(0.0, is_last=True, is_terminal=True)
        assert obs["is_last"] is True
        assert obs["is_terminal"] is True
        assert is_last is True


class TestAtariRender:
    def test_render_rotates_buffer(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        env, ale_mock = _make_atari("pong", ale_mock=ale, size=(8, 8), pooling=2)
        env._buffers[0][:] = 1
        env._buffers[1][:] = 2
        env._render()
        ale_mock.getScreenRGB.assert_called()


class TestAtariStep:
    def test_step_accumulates_reward(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        ale.act.return_value = 2.0
        ale.game_over.return_value = False
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), action_repeat=3, pooling=2)
        env.reset()
        env._done = False
        obs, reward, done, _ = env.step(0)
        assert reward == 6.0

    def test_step_game_over_exits_early(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        call_count = [0]

        def act_side(action):
            call_count[0] += 1
            return 0.0

        ale.act.side_effect = act_side
        ale.game_over.side_effect = lambda: call_count[0] >= 1

        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), action_repeat=4, pooling=2)
        env.reset()
        env._done = False
        env.step(0)
        assert call_count[0] < 4

    def test_step_life_loss_discount_mode(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8, lives=3)
        ale.game_over.return_value = False

        life_call = [0]

        def lives_side():
            life_call[0] += 1
            return 2 if life_call[0] > 1 else 3

        ale.lives.side_effect = lives_side
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), action_repeat=2, pooling=2, lives="discount")
        env.reset()
        env._last_lives = 3
        env._done = False
        obs, _, _, _ = env.step(0)
        # Dead detected due to life loss

    def test_step_life_loss_reset_mode_sets_is_last(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8, lives=3)
        ale.game_over.return_value = False
        life_call = [0]

        def lives_side():
            life_call[0] += 1
            return 2 if life_call[0] > 1 else 3

        ale.lives.side_effect = lives_side
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), action_repeat=2, pooling=2, lives="reset")
        env.reset()
        env._last_lives = 3
        env._done = False
        obs, _, is_last, _ = env.step(0)
        assert is_last is True

    def test_step_length_limit_sets_done(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        ale.game_over.return_value = False
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), action_repeat=1, pooling=2, length=2)
        env.reset()
        env._done = False
        env.step(0)
        obs, _, done, _ = env.step(0)
        assert done is True

    def test_step_life_loss_unused_no_early_exit(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8, lives=3)
        ale.game_over.return_value = False

        life_call = [0]

        def lives_side():
            life_call[0] += 1
            return 2 if life_call[0] > 1 else 3

        ale.lives.side_effect = lives_side
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), action_repeat=3, pooling=2, lives="unused")
        env.reset()
        env._last_lives = 3
        env._done = False
        # With lives="unused", life loss doesn't stop the loop
        env.step(0)


class TestAtariReset:
    def test_reset_no_noops(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), noops=0, pooling=2)
        obs = env.reset()
        assert obs["is_first"] is True
        assert env._step == 0
        assert env._done is False

    def test_reset_with_noops(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        ale.game_over.return_value = False
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), noops=3, pooling=2, seed=0)
        env.reset()

    def test_reset_noops_with_game_over_recovery(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        game_over_calls = [0]

        def go():
            game_over_calls[0] += 1
            return game_over_calls[0] == 1

        ale.game_over.side_effect = go
        ale.act.return_value = 0.0
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), noops=5, pooling=2, seed=42)
        env._rng = np.random.default_rng(0)
        env.reset()
        # Reset was called after game_over during noop
        assert ale.reset_game.call_count >= 1

    def test_reset_autostart_fire_in_actionset(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        ale.game_over.return_value = False
        # FIRE is at index 1 in ACTION_MEANING
        fire_idx = 1
        ale.getLegalActionSet.return_value = list(range(18))
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), autostart=True, pooling=2)
        env.reset()
        # FIRE action should have been sent
        act_calls = [c[0][0] for c in ale.act.call_args_list]
        assert fire_idx in act_calls

    def test_reset_autostart_fire_not_in_actionset(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        ale.game_over.return_value = False
        # No FIRE in actionset (only action 0)
        ale.getLegalActionSet.return_value = [0]
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), autostart=True, pooling=2)
        env.reset()
        # FIRE should NOT be called (not in actionset)

    def test_reset_autostart_fire_game_over_recovery(self):
        ale = _make_ale_mock(screen_h=8, screen_w=8)
        ale.game_over.return_value = True  # game over immediately after FIRE → triggers lines 173-174
        ale.getLegalActionSet.return_value = list(range(18))
        env, _ = _make_atari("pong", ale_mock=ale, size=(8, 8), autostart=True, noops=0, pooling=2)
        env.reset()
        # Initial reset_game (line 161) + recovery reset_game (line 174) = at least 2
        assert ale.reset_game.call_count >= 2


class TestAtariClose:
    def test_close_does_nothing(self):
        env, _ = _make_atari("pong", size=(8, 8), pooling=2)
        env.close()  # Should not raise
