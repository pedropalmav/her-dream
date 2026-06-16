from unittest.mock import MagicMock, patch

import pytest

# dmc_subtle uses top-level imports from dm_control + lxml.
# dm_control is installed, so we import the module directly and patch
# the specific Physics/Environment constructors to skip actual physics.
import envs.dmc_subtle as subtle

# ---------------------------------------------------------------------------
# Minimal XML snippets for _modify_xml_element_size
# ---------------------------------------------------------------------------

SIMPLE_XML = b"""<mujoco>
  <worldbody>
    <geom name="target" size="0.05"/>
    <geom name="pointmass" size="0.01"/>
  </worldbody>
</mujoco>"""

BALL_IN_CUP_XML = b"""<mujoco>
  <worldbody>
    <geom name="ball" size="0.025"/>
  </worldbody>
  <tendon>
    <spatial name="string" width="0.003"/>
  </tendon>
</mujoco>"""

CARTPOLE_XML = b"""<mujoco>
  <default>
    <default class="pole">
      <geom size="0.02"/>
    </default>
  </default>
</mujoco>"""


# ---------------------------------------------------------------------------
# _modify_xml_element_size
# ---------------------------------------------------------------------------


class TestModifyXmlElementSize:
    def test_modifies_geom_size(self):
        result = subtle._modify_xml_element_size(SIMPLE_XML, "target", "0.99")
        assert b"0.99" in result

    def test_element_not_found_raises_value_error(self):
        with pytest.raises(ValueError, match="nonexistent"):
            subtle._modify_xml_element_size(SIMPLE_XML, "nonexistent", "0.1")

    def test_custom_element_type(self):
        xml = b"""<mujoco><body name="mybody" size="1.0"/></mujoco>"""
        result = subtle._modify_xml_element_size(xml, "mybody", "2.0", element_type="body")
        assert b"2.0" in result

    def test_returns_bytes(self):
        result = subtle._modify_xml_element_size(SIMPLE_XML, "target", "0.1")
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# Helpers: mock Physics + Environment + get_model_and_assets for each module
# ---------------------------------------------------------------------------


def _mock_env():
    return MagicMock()


# ---------------------------------------------------------------------------
# reacher_subtle
# ---------------------------------------------------------------------------


class TestReacherSubtle:
    def test_creates_control_environment(self):
        mock_physics = MagicMock()
        mock_task = MagicMock()
        mock_ce = MagicMock()
        with (
            patch("dm_control.suite.reacher.Physics.from_xml_string", return_value=mock_physics),
            patch("dm_control.suite.reacher.Reacher", return_value=mock_task),
            patch("dm_control.rl.control.Environment", return_value=mock_ce) as mock_env_cls,
        ):
            result = subtle.reacher_subtle(random=0)
        mock_env_cls.assert_called_once()
        assert result is mock_ce

    def test_applies_scale_to_target_size(self):
        with (
            patch("dm_control.suite.reacher.Physics.from_xml_string"),
            patch("dm_control.suite.reacher.Reacher") as mock_reacher,
            patch("dm_control.rl.control.Environment"),
        ):
            subtle.reacher_subtle(random=42)
        call_kwargs = mock_reacher.call_args[1]
        from dm_control.suite import reacher

        expected = reacher._SMALL_TARGET * subtle.SCALES["reacher_subtle"]
        assert abs(call_kwargs["target_size"] - expected) < 1e-10

    def test_environment_kwargs_none_defaults_to_empty(self):
        with (
            patch("dm_control.suite.reacher.Physics.from_xml_string"),
            patch("dm_control.suite.reacher.Reacher"),
            patch("dm_control.rl.control.Environment") as mock_env_cls,
        ):
            subtle.reacher_subtle(environment_kwargs=None)
        mock_env_cls.assert_called_once()

    def test_environment_kwargs_passed_through(self):
        with (
            patch("dm_control.suite.reacher.Physics.from_xml_string"),
            patch("dm_control.suite.reacher.Reacher"),
            patch("dm_control.rl.control.Environment") as mock_env_cls,
        ):
            subtle.reacher_subtle(environment_kwargs={"flat_observation": True})
        call_kwargs = mock_env_cls.call_args[1]
        assert call_kwargs.get("flat_observation") is True


# ---------------------------------------------------------------------------
# finger_turn_subtle
# ---------------------------------------------------------------------------


class TestFingerTurnSubtle:
    def test_creates_control_environment(self):
        with (
            patch("dm_control.suite.finger.Physics.from_xml_string"),
            patch("dm_control.suite.finger.Turn"),
            patch("dm_control.rl.control.Environment") as mock_env_cls,
        ):
            subtle.finger_turn_subtle(random=0)
        mock_env_cls.assert_called_once()

    def test_applies_scale_to_target_radius(self):
        with (
            patch("dm_control.suite.finger.Physics.from_xml_string"),
            patch("dm_control.suite.finger.Turn") as mock_turn,
            patch("dm_control.rl.control.Environment"),
        ):
            subtle.finger_turn_subtle(random=0)
        call_kwargs = mock_turn.call_args[1]
        from dm_control.suite import finger

        expected = finger._HARD_TARGET_SIZE * subtle.SCALES["finger_turn_subtle"]
        assert abs(call_kwargs["target_radius"] - expected) < 1e-10

    def test_environment_kwargs_passthrough(self):
        with (
            patch("dm_control.suite.finger.Physics.from_xml_string"),
            patch("dm_control.suite.finger.Turn"),
            patch("dm_control.rl.control.Environment") as mock_env_cls,
        ):
            subtle.finger_turn_subtle(environment_kwargs={"n_sub_steps": 2})
        call_kwargs = mock_env_cls.call_args[1]
        assert call_kwargs.get("n_sub_steps") == 2


# ---------------------------------------------------------------------------
# point_mass_subtle
# ---------------------------------------------------------------------------


class TestPointMassSubtle:
    def test_creates_control_environment(self):
        with (
            patch("dm_control.suite.point_mass.Physics.from_xml_string"),
            patch("dm_control.suite.point_mass.PointMass"),
            patch("dm_control.rl.control.Environment") as mock_env_cls,
            patch("dm_control.suite.point_mass.get_model_and_assets", return_value=(SIMPLE_XML, {})),
        ):
            subtle.point_mass_subtle(random=0)
        mock_env_cls.assert_called_once()

    def test_environment_kwargs_passthrough(self):
        with (
            patch("dm_control.suite.point_mass.Physics.from_xml_string"),
            patch("dm_control.suite.point_mass.PointMass"),
            patch("dm_control.rl.control.Environment") as mock_env_cls,
            patch("dm_control.suite.point_mass.get_model_and_assets", return_value=(SIMPLE_XML, {})),
        ):
            subtle.point_mass_subtle(environment_kwargs={"n_sub_steps": 3})
        call_kwargs = mock_env_cls.call_args[1]
        assert call_kwargs.get("n_sub_steps") == 3


# ---------------------------------------------------------------------------
# ball_in_cup_catch_subtle
# ---------------------------------------------------------------------------


class TestBallInCupCatchSubtle:
    def test_creates_control_environment(self):
        with (
            patch("dm_control.suite.ball_in_cup.Physics.from_xml_string"),
            patch("dm_control.suite.ball_in_cup.BallInCup"),
            patch("dm_control.rl.control.Environment") as mock_env_cls,
            patch("dm_control.suite.ball_in_cup.get_model_and_assets", return_value=(BALL_IN_CUP_XML, {})),
        ):
            subtle.ball_in_cup_catch_subtle(random=0)
        mock_env_cls.assert_called_once()

    def test_environment_kwargs_passthrough(self):
        with (
            patch("dm_control.suite.ball_in_cup.Physics.from_xml_string"),
            patch("dm_control.suite.ball_in_cup.BallInCup"),
            patch("dm_control.rl.control.Environment") as mock_env_cls,
            patch("dm_control.suite.ball_in_cup.get_model_and_assets", return_value=(BALL_IN_CUP_XML, {})),
        ):
            subtle.ball_in_cup_catch_subtle(environment_kwargs={"flat_observation": True})
        call_kwargs = mock_env_cls.call_args[1]
        assert call_kwargs.get("flat_observation") is True

    def test_missing_ball_raises_value_error(self):
        no_ball_xml = b"""<mujoco><worldbody></worldbody>
  <tendon><spatial name="string" width="0.003"/></tendon></mujoco>"""
        with (
            patch("dm_control.suite.ball_in_cup.get_model_and_assets", return_value=(no_ball_xml, {})),
            pytest.raises(ValueError, match="ball"),
        ):
            subtle.ball_in_cup_catch_subtle()

    def test_missing_string_raises_value_error(self):
        no_string_xml = b"""<mujoco><worldbody><geom name="ball" size="0.025"/></worldbody>
  <tendon></tendon></mujoco>"""
        with (
            patch("dm_control.suite.ball_in_cup.get_model_and_assets", return_value=(no_string_xml, {})),
            pytest.raises(ValueError, match="string"),
        ):
            subtle.ball_in_cup_catch_subtle()


# ---------------------------------------------------------------------------
# _get_cartpole_subtle_physics + cartpole_swingup_subtle
# ---------------------------------------------------------------------------


class TestCartpoleSubtle:
    def test_creates_control_environment(self):
        with (
            patch("dm_control.suite.cartpole.Physics.from_xml_string"),
            patch("dm_control.suite.cartpole.Balance"),
            patch("dm_control.rl.control.Environment") as mock_env_cls,
            patch("dm_control.suite.cartpole.get_model_and_assets", return_value=(CARTPOLE_XML, {})),
        ):
            subtle.cartpole_swingup_subtle(random=0)
        mock_env_cls.assert_called_once()

    def test_applies_scale_to_pole_radius(self):
        with (
            patch("dm_control.suite.cartpole.Physics.from_xml_string") as mock_phys,
            patch("dm_control.suite.cartpole.Balance"),
            patch("dm_control.rl.control.Environment"),
            patch("dm_control.suite.cartpole.get_model_and_assets", return_value=(CARTPOLE_XML, {})),
        ):
            subtle.cartpole_swingup_subtle()
        # The physics was created from modified XML
        mock_phys.assert_called_once()
        modified_xml = mock_phys.call_args[0][0]
        # Original size 0.02 * (1/20) = 0.001
        assert b"0.001" in modified_xml

    def test_environment_kwargs_passthrough(self):
        with (
            patch("dm_control.suite.cartpole.Physics.from_xml_string"),
            patch("dm_control.suite.cartpole.Balance"),
            patch("dm_control.rl.control.Environment") as mock_env_cls,
            patch("dm_control.suite.cartpole.get_model_and_assets", return_value=(CARTPOLE_XML, {})),
        ):
            subtle.cartpole_swingup_subtle(environment_kwargs={"n_sub_steps": 4})
        call_kwargs = mock_env_cls.call_args[1]
        assert call_kwargs.get("n_sub_steps") == 4

    def test_missing_default_pole_geom_raises(self):
        no_pole_xml = b"""<mujoco><default></default></mujoco>"""
        with (
            patch("dm_control.suite.cartpole.get_model_and_assets", return_value=(no_pole_xml, {})),
            pytest.raises(ValueError, match="pole"),
        ):
            subtle.cartpole_swingup_subtle()

    def test_missing_size_attr_raises(self):
        no_size_xml = b"""<mujoco>
  <default>
    <default class="pole">
      <geom name="pole_geom"/>
    </default>
  </default>
</mujoco>"""
        with (
            patch("dm_control.suite.cartpole.get_model_and_assets", return_value=(no_size_xml, {})),
            pytest.raises(ValueError, match="size"),
        ):
            subtle.cartpole_swingup_subtle()
