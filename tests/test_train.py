import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import her_dream.tools as tools
import train


def make_config(tmp_path, **overrides):
    defaults = dict(
        seed=42,
        deterministic_run=False,
        logdir=str(tmp_path / "logdir"),
        logger=SimpleNamespace(backends=["file"]),
        env=SimpleNamespace(goal_sample="buffer"),
        mission_text=False,
        model=SimpleNamespace(),
        device="cpu",
        trainer=SimpleNamespace(),
        # Mode-selecting flags (default = from-scratch training).
        load_from=None,
        freeze_wm=False,
        wm_only=False,
        train_text_only=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


# ---------------------------------------------------------------------------
# Individual mock fixtures — each patches exactly one external dependency.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_seed():
    with patch("train.tools.set_seed_everywhere") as m:
        yield m


@pytest.fixture
def mock_enable_det():
    with patch("train.tools.enable_deterministic_run") as m:
        yield m


@pytest.fixture
def mock_console_log():
    with patch("train.tools.setup_console_log") as m:
        yield m


@pytest.fixture
def mock_make_logger():
    with patch("train.tools.make_logger") as m:
        yield m


@pytest.fixture
def mock_collect_optim():
    with patch("train.tools.recursively_collect_optim_state_dict") as m:
        yield m


@pytest.fixture
def mock_make_envs():
    with patch("train.make_envs") as m:
        m.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        yield m


@pytest.fixture
def mock_make_reward():
    with patch("train.make_reward") as m:
        yield m


@pytest.fixture
def mock_make_buffer():
    with patch("train.make_buffer") as m:
        yield m


@pytest.fixture
def mock_dreamer_cls():
    with patch("train.Dreamer") as m:
        agent = m.return_value
        agent.to.return_value = agent  # .to() returns self
        # `load_checkpoint` sums numel() over parameters(); keep it iterable.
        agent.parameters.return_value = []
        # Default to "no extra freeze applied" so load_checkpoint branches are
        # only taken when a test sets them explicitly.
        agent.train_text_only = False
        agent.freeze_wm = False
        yield m


@pytest.fixture
def mock_trainer_cls():
    with patch("train.OnlineTrainer") as m:
        yield m


@pytest.fixture
def mock_torch_save():
    with patch("torch.save") as m:
        yield m


@pytest.fixture
def mock_torch_load():
    with patch("train.torch.load") as m:
        m.return_value = {"agent_state_dict": {}}
        yield m


@pytest.fixture
def mock_atexit():
    with patch("atexit.register") as m:
        yield m


# ---------------------------------------------------------------------------
# Composed fixture — activates all patches so main.__wrapped__ runs safely.
# pytest deduplicates fixtures, so requesting both `noop` and e.g.
# `mock_seed` in the same test gives the same mock instance to both.
# ---------------------------------------------------------------------------


@pytest.fixture
def noop(
    mock_seed,
    mock_enable_det,
    mock_console_log,
    mock_make_logger,
    mock_collect_optim,
    mock_make_envs,
    mock_make_reward,
    mock_make_buffer,
    mock_dreamer_cls,
    mock_trainer_cls,
    mock_torch_save,
    mock_torch_load,
    mock_atexit,
):
    """Activate all patches so main.__wrapped__ can run without side effects."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEntryPoint:
    def test_main_called_when_run_as_script(self, monkeypatch):
        """The __main__ guard calls main().

        runpy re-executes the module in a fresh namespace, so patching
        train.main directly has no effect. Instead we patch hydra.main with a
        no-op decorator so the redefined `main` becomes a harmless callable,
        and coverage records line 72 as executed.
        """
        import runpy

        import hydra

        monkeypatch.setattr(sys, "argv", ["train.py"])
        monkeypatch.setattr(hydra, "main", lambda **kw: lambda fn: lambda: None)
        runpy.run_module("train", run_name="__main__")


class TestSeedAndDeterminism:
    def test_sets_seed(self, noop, mock_seed, cfg):
        train.main.__wrapped__(cfg)
        mock_seed.assert_called_once_with(cfg.seed)

    def test_no_deterministic_run(self, noop, mock_enable_det, cfg):
        cfg.deterministic_run = False
        train.main.__wrapped__(cfg)
        mock_enable_det.assert_not_called()

    def test_deterministic_run_enabled(self, noop, mock_enable_det, cfg):
        cfg.deterministic_run = True
        train.main.__wrapped__(cfg)
        mock_enable_det.assert_called_once()


class TestLogdirSetup:
    def test_creates_logdir(self, noop, tmp_path):
        logdir = tmp_path / "new" / "nested" / "logdir"
        cfg = make_config(tmp_path, logdir=str(logdir))
        train.main.__wrapped__(cfg)
        assert logdir.exists()

    def test_sets_up_console_log(self, noop, mock_console_log, cfg):
        train.main.__wrapped__(cfg)
        mock_console_log.assert_called_once_with(pathlib.Path(cfg.logdir).expanduser(), filename="console.log")

    def test_registers_atexit_to_close_console(self, noop, mock_atexit, mock_console_log, cfg):
        train.main.__wrapped__(cfg)
        mock_atexit.assert_called_once()
        registered_fn = mock_atexit.call_args[0][0]
        registered_fn()
        mock_console_log.return_value.close.assert_called_once()


class TestLogger:
    def test_creates_logger(self, noop, mock_make_logger, cfg):
        train.main.__wrapped__(cfg)
        mock_make_logger.assert_called_once_with(cfg.logger, pathlib.Path(cfg.logdir).expanduser())

    def test_logs_hydra_config(self, noop, mock_make_logger, cfg):
        train.main.__wrapped__(cfg)
        mock_make_logger.return_value.log_hydra_config.assert_called_once_with(cfg)


class TestGoalSampleAssertion:
    def test_buffer_goal_sample_skips_assertion(self, noop, cfg):
        cfg.env.goal_sample = "buffer"
        cfg.mission_text = False
        train.main.__wrapped__(cfg)  # must not raise

    def test_text_goal_sample_with_mission_text_passes(self, noop, cfg):
        cfg.env.goal_sample = "text"
        cfg.mission_text = True
        train.main.__wrapped__(cfg)  # must not raise

    def test_text_goal_sample_without_mission_text_raises(
        self, mock_seed, mock_enable_det, mock_console_log, mock_atexit, mock_make_logger, cfg
    ):
        cfg.env.goal_sample = "text"
        cfg.mission_text = False
        with pytest.raises(AssertionError):
            train.main.__wrapped__(cfg)


class TestFactories:
    def test_creates_envs(self, noop, mock_make_envs, cfg):
        train.main.__wrapped__(cfg)
        mock_make_envs.assert_called_once_with(cfg.env)

    def test_creates_reward_function(self, noop, mock_make_reward, cfg):
        train.main.__wrapped__(cfg)
        mock_make_reward.assert_called_once_with(cfg)

    def test_creates_buffer_with_reward(self, noop, mock_make_buffer, mock_make_reward, cfg):
        train.main.__wrapped__(cfg)
        mock_make_buffer.assert_called_once_with(cfg, mock_make_reward.return_value)


class TestAgentAndTrainer:
    def test_creates_dreamer_with_correct_args(self, noop, mock_dreamer_cls, mock_make_envs, mock_make_reward, cfg):
        train.main.__wrapped__(cfg)
        _, _, obs_space, act_space = mock_make_envs.return_value
        mock_dreamer_cls.assert_called_once_with(
            cfg.model, obs_space, act_space, reward_function=mock_make_reward.return_value
        )

    def test_moves_agent_to_device(self, noop, mock_dreamer_cls, cfg):
        train.main.__wrapped__(cfg)
        mock_dreamer_cls.return_value.to.assert_called_once_with(cfg.device)

    def test_sets_rssm_on_buffer(self, noop, mock_make_buffer, mock_dreamer_cls, cfg):
        train.main.__wrapped__(cfg)
        assert mock_make_buffer.return_value.rssm is mock_dreamer_cls.return_value.rssm

    def test_creates_trainer_with_correct_args(
        self,
        noop,
        mock_trainer_cls,
        mock_make_buffer,
        mock_make_logger,
        mock_make_envs,
        mock_make_reward,
        cfg,
    ):
        train.main.__wrapped__(cfg)
        train_envs, eval_envs, _, _ = mock_make_envs.return_value
        mock_trainer_cls.assert_called_once_with(
            cfg.trainer,
            mock_make_buffer.return_value,
            mock_make_logger.return_value,
            pathlib.Path(cfg.logdir).expanduser(),
            train_envs,
            eval_envs,
            reward_function=mock_make_reward.return_value,
        )

    def test_begins_training(self, noop, mock_trainer_cls, mock_dreamer_cls, cfg):
        train.main.__wrapped__(cfg)
        mock_trainer_cls.return_value.begin.assert_called_once_with(mock_dreamer_cls.return_value)


class TestCheckpointSave:
    def test_saves_checkpoint(self, noop, mock_torch_save, cfg):
        train.main.__wrapped__(cfg)
        mock_torch_save.assert_called_once()

    def test_checkpoint_saved_to_logdir(self, noop, mock_torch_save, cfg):
        train.main.__wrapped__(cfg)
        _, save_path = mock_torch_save.call_args[0]
        assert save_path == pathlib.Path(cfg.logdir).expanduser() / "latest.pt"

    def test_checkpoint_contains_agent_state(self, noop, mock_torch_save, mock_dreamer_cls, cfg):
        train.main.__wrapped__(cfg)
        saved_dict, _ = mock_torch_save.call_args[0]
        assert "agent_state_dict" in saved_dict
        assert saved_dict["agent_state_dict"] is mock_dreamer_cls.return_value.state_dict()

    def test_checkpoint_contains_optim_state(self, noop, mock_torch_save, mock_collect_optim, mock_dreamer_cls, cfg):
        train.main.__wrapped__(cfg)
        saved_dict, _ = mock_torch_save.call_args[0]
        assert "optims_state_dict" in saved_dict
        mock_collect_optim.assert_called_once_with(mock_dreamer_cls.return_value)


# ---------------------------------------------------------------------------
# Loading from a checkpoint: post-training and text-distillation modes.
# ---------------------------------------------------------------------------


def make_ckpt(tmp_path, name="src"):
    """Create a source logdir containing a latest.pt file and return its path."""
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    (src / "latest.pt").write_bytes(b"")  # torch.load is mocked; just needs to exist
    return src


class TestNoLoadFrom:
    def test_does_not_load_when_load_from_none(self, noop, mock_torch_load, mock_dreamer_cls, cfg):
        cfg.load_from = None
        train.main.__wrapped__(cfg)
        mock_torch_load.assert_not_called()
        mock_dreamer_cls.return_value.load_state_dict.assert_not_called()


class TestPostTrainMode:
    def test_loads_checkpoint_from_load_from(self, noop, mock_torch_load, mock_dreamer_cls, tmp_path):
        src = make_ckpt(tmp_path)
        cfg = make_config(tmp_path, load_from=str(src), freeze_wm=True)
        mock_dreamer_cls.return_value.freeze_wm = True
        train.main.__wrapped__(cfg)
        mock_torch_load.assert_called_once()
        assert mock_torch_load.call_args[0][0] == src / "latest.pt"

    def test_loads_state_dict_and_reclones(self, noop, mock_torch_load, mock_dreamer_cls, tmp_path):
        src = make_ckpt(tmp_path)
        cfg = make_config(tmp_path, load_from=str(src), freeze_wm=True)
        agent = mock_dreamer_cls.return_value
        agent.freeze_wm = True
        train.main.__wrapped__(cfg)
        agent.load_state_dict.assert_called_once_with(mock_torch_load.return_value["agent_state_dict"])
        agent.clone_and_freeze.assert_called_once()

    def test_applies_freeze_wm_when_frozen(self, noop, mock_torch_load, mock_dreamer_cls, tmp_path):
        src = make_ckpt(tmp_path)
        cfg = make_config(tmp_path, load_from=str(src), freeze_wm=True)
        agent = mock_dreamer_cls.return_value
        agent.freeze_wm = True
        agent.train_text_only = False
        train.main.__wrapped__(cfg)
        agent._apply_freeze_wm.assert_called_once()
        agent._apply_train_text_only.assert_not_called()

    def test_no_freeze_apply_when_not_frozen(self, noop, mock_torch_load, mock_dreamer_cls, tmp_path):
        src = make_ckpt(tmp_path)
        cfg = make_config(tmp_path, load_from=str(src), freeze_wm=False)
        agent = mock_dreamer_cls.return_value
        agent.freeze_wm = False
        agent.train_text_only = False
        train.main.__wrapped__(cfg)  # full fine-tune; just warns
        agent._apply_freeze_wm.assert_not_called()
        agent._apply_train_text_only.assert_not_called()

    def test_missing_checkpoint_raises(self, noop, mock_dreamer_cls, tmp_path):
        cfg = make_config(tmp_path, load_from=str(tmp_path / "does_not_exist"), freeze_wm=True)
        with pytest.raises(FileNotFoundError):
            train.main.__wrapped__(cfg)

    def test_wm_only_with_load_from_raises(self, mock_seed, mock_enable_det, mock_console_log, mock_atexit, tmp_path):
        cfg = make_config(tmp_path, load_from=str(make_ckpt(tmp_path)), wm_only=True)
        with pytest.raises(ValueError):
            train.main.__wrapped__(cfg)


class TestResetActorCritic:
    """`_world_model_state_dict` keeps the WM and drops only actor/critic.

    Used to post-train on a vanilla-Dreamer checkpoint whose actor/critic were
    built without a goal input. The point of the helper is that it is *not* a
    blanket strict=False: a partial load of the world model must raise.
    """

    @staticmethod
    def agent_with(shapes):
        agent = SimpleNamespace()
        agent.state_dict = lambda: {k: torch.zeros(v) for k, v in shapes.items()}
        return agent

    def test_keeps_world_model_and_drops_actor_critic(self):
        agent = self.agent_with({"rssm.w": (4, 4), "encoder.w": (2,), "ac.actor.mlp.w": (3, 8)})
        ckpt = {"rssm.w": torch.ones(4, 4), "encoder.w": torch.ones(2), "ac.actor.mlp.w": torch.ones(3, 5)}
        kept = train._world_model_state_dict(agent, ckpt)
        assert set(kept) == {"rssm.w", "encoder.w"}

    def test_drops_the_return_ema_with_the_actor_critic(self):
        # The return normalizer lives on the ActorCritic, so resetting the policy
        # resets its return scale too rather than carrying a stale one over.
        agent = self.agent_with({"rssm.w": (4, 4), "ac.return_ema.ema_vals": (2,)})
        ckpt = {"rssm.w": torch.ones(4, 4), "ac.return_ema.ema_vals": torch.ones(2)}
        assert set(train._world_model_state_dict(agent, ckpt)) == {"rssm.w"}

    def test_legacy_flat_checkpoint_is_migrated_before_the_split(self):
        # A pre-`ActorCritic` checkpoint uses flat keys; once migrated it splits
        # exactly like a current one instead of tripping the "no counterpart" guard.
        agent = self.agent_with({"rssm.w": (4, 4), "ac.actor.mlp.w": (3, 8)})
        legacy = {"rssm.w": torch.ones(4, 4), "actor.mlp.w": torch.ones(3, 8), "threshold_onehot": torch.ones(3)}
        migrated = tools.migrate_agent_state_dict(legacy)
        assert set(migrated) == {"rssm.w", "ac.actor.mlp.w", "ac.threshold_onehot"}
        assert set(train._world_model_state_dict(agent, migrated)) == {"rssm.w"}

    def test_drops_heads_absent_from_the_agent(self):
        # Checkpoint trained with the heads, agent built without them.
        agent = self.agent_with({"rssm.w": (4, 4)})
        ckpt = {"rssm.w": torch.ones(4, 4), "reward.last.weight": torch.ones(2), "cont.last.bias": torch.ones(1)}
        assert set(train._world_model_state_dict(agent, ckpt)) == {"rssm.w"}

    def test_newly_enabled_heads_stay_at_init(self, capsys):
        # The mirror case: heads switched on for a post-training run whose source
        # checkpoint predates them. They start fresh instead of failing the load.
        agent = self.agent_with({
            "rssm.w": (4, 4),
            "reward.last.weight": (2,),
            "cont.last.bias": (1,),
            "_frozen_reward.last.weight": (2,),
        })
        kept = train._world_model_state_dict(agent, {"rssm.w": torch.ones(4, 4)})
        assert set(kept) == {"rssm.w"}
        assert "fresh init" in capsys.readouterr().out

    def test_missing_non_head_tensor_still_raises_when_heads_enabled(self):
        # The head exemption must not weaken the partial-load guard.
        agent = self.agent_with({"rssm.w": (4, 4), "encoder.w": (2,), "reward.last.weight": (2,)})
        with pytest.raises(ValueError, match="absent from the checkpoint"):
            train._world_model_state_dict(agent, {"rssm.w": torch.ones(4, 4)})

    def test_unknown_extra_key_raises(self):
        agent = self.agent_with({"rssm.w": (4, 4)})
        ckpt = {"rssm.w": torch.ones(4, 4), "text_encoder.gru.weight": torch.ones(2)}
        with pytest.raises(ValueError, match="no counterpart"):
            train._world_model_state_dict(agent, ckpt)

    def test_world_model_shape_mismatch_raises(self):
        agent = self.agent_with({"rssm.w": (4, 4)})
        with pytest.raises(ValueError, match="Shape mismatch"):
            train._world_model_state_dict(agent, {"rssm.w": torch.ones(8, 8)})

    def test_missing_world_model_tensor_raises(self):
        agent = self.agent_with({"rssm.w": (4, 4), "encoder.w": (2,)})
        with pytest.raises(ValueError, match="absent from the checkpoint"):
            train._world_model_state_dict(agent, {"rssm.w": torch.ones(4, 4)})

    def test_requires_load_from(self, tmp_path):
        cfg = make_config(tmp_path)
        cfg.reset_actor_critic = True
        cfg.load_from = None
        with pytest.raises(ValueError, match="only makes sense together with load_from"):
            train.validate_config(cfg)


class TestPlan2ExploreCheckpoints:
    """A pretraining checkpoint carries modules the achiever does not build."""

    @staticmethod
    def agent_with(shapes):
        agent = SimpleNamespace()
        agent.state_dict = lambda: {k: torch.zeros(v) for k, v in shapes.items()}
        return agent

    def test_drops_the_explorer_and_ensemble_on_a_strict_load(self, capsys):
        # This is the phase-2 handoff: pretrain with plan2explore, then post-train
        # the achiever on the frozen world model.
        agent = self.agent_with({"rssm.w": (4, 4), "ac.actor.w": (3, 8)})
        ckpt = {
            "rssm.w": torch.ones(4, 4),
            "ac.actor.w": torch.ones(3, 8),
            "explorer.actor.w": torch.ones(3, 4),
            "disag.heads.0.last.weight": torch.ones(2),
        }
        kept = train._drop_optional_absent(agent, ckpt)
        assert set(kept) == {"rssm.w", "ac.actor.w"}
        assert "explorer" in capsys.readouterr().out

    def test_keeps_everything_the_agent_does_build(self):
        agent = self.agent_with({"rssm.w": (4, 4), "explorer.actor.w": (3, 4)})
        ckpt = {"rssm.w": torch.ones(4, 4), "explorer.actor.w": torch.ones(3, 4)}
        assert set(train._drop_optional_absent(agent, ckpt)) == set(ckpt)

    def test_an_unknown_extra_key_is_left_for_load_state_dict_to_reject(self):
        # Only *known* optional modules are dropped; strictness is preserved.
        agent = self.agent_with({"rssm.w": (4, 4)})
        ckpt = {"rssm.w": torch.ones(4, 4), "mystery.w": torch.ones(2)}
        assert "mystery.w" in train._drop_optional_absent(agent, ckpt)

    def test_rejects_load_from(self, tmp_path):
        cfg = make_config(tmp_path, load_from=str(make_ckpt(tmp_path)))
        cfg.plan2explore = True
        with pytest.raises(ValueError, match="does not take load_from"):
            train.validate_config(cfg)

    @pytest.mark.parametrize("flag", ["wm_only", "freeze_wm", "train_text_only"])
    def test_rejects_the_other_modes(self, tmp_path, flag):
        cfg = make_config(tmp_path)
        cfg.plan2explore = True
        setattr(cfg, flag, True)
        cfg.mission_text = True
        with pytest.raises(ValueError, match="exclusive with wm_only"):
            train.validate_config(cfg)


class TestDistillMode:
    def test_applies_train_text_only(self, noop, mock_torch_load, mock_dreamer_cls, tmp_path):
        src = make_ckpt(tmp_path)
        cfg = make_config(tmp_path, load_from=str(src), train_text_only=True, mission_text=True)
        agent = mock_dreamer_cls.return_value
        agent.train_text_only = True
        train.main.__wrapped__(cfg)
        agent._apply_train_text_only.assert_called_once()
        agent._apply_freeze_wm.assert_not_called()

    def test_forces_off_incompatible_modes(self, noop, mock_torch_load, mock_dreamer_cls, tmp_path):
        src = make_ckpt(tmp_path)
        cfg = make_config(
            tmp_path, load_from=str(src), train_text_only=True, mission_text=True, wm_only=True, freeze_wm=True
        )
        mock_dreamer_cls.return_value.train_text_only = True
        train.main.__wrapped__(cfg)
        assert cfg.wm_only is False
        assert cfg.freeze_wm is False

    def test_requires_load_from(self, mock_seed, mock_enable_det, mock_console_log, mock_atexit, tmp_path):
        cfg = make_config(tmp_path, train_text_only=True, mission_text=True, load_from=None)
        with pytest.raises(ValueError):
            train.main.__wrapped__(cfg)

    def test_requires_mission_text(self, mock_seed, mock_enable_det, mock_console_log, mock_atexit, tmp_path):
        cfg = make_config(tmp_path, train_text_only=True, mission_text=False, load_from=str(make_ckpt(tmp_path)))
        with pytest.raises(ValueError):
            train.main.__wrapped__(cfg)
