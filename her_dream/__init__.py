"""her_dream — goal-conditioned R2-Dreamer (HER buffer + Dreamer world model).

Public API re-exports for ergonomic use once the package is installed.
"""

from her_dream.actor_critic import ActorCritic
from her_dream.buffers import make_buffer
from her_dream.dreamer import Dreamer
from her_dream.envs import make_env, make_envs
from her_dream.goals import goal_from_latent, make_goal_spec, reward_state
from her_dream.plan2explore import Disagreement
from her_dream.rewards import make_reward
from her_dream.temporal_distance import TemporalDistance
from her_dream.trainer import OnlineTrainer

__all__ = [
    "Dreamer",
    "ActorCritic",
    "Disagreement",
    "TemporalDistance",
    "OnlineTrainer",
    "make_buffer",
    "make_env",
    "make_envs",
    "make_reward",
    "make_goal_spec",
    "goal_from_latent",
    "reward_state",
]
