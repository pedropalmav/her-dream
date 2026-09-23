"""The BFS-oracle z-check: is the goal-z attainable at all?

Drives a shortest-path plan to the target pose for real and compares the
posterior z there against the goal z. If the oracle — which provably stands on
the target state — cannot reproduce the goal z, then no policy can, and any
policy failure measured against that goal says nothing about the policy. The
control is the *sampling floor*: a second draw from the same posterior logits,
i.e. the irreducible Gumbel noise, an upper bound on any exact-match rate.
"""

from .envs import agent_pose, unwrap_env
from .metrics import groups_matched, reward_of
from .rollout import posterior_rollout, scripted_policy


def oracle_z_check(agent, spec, env, goal, target_pos, target_dir, plan, device, seed=None) -> dict:
    """Run `plan` to the target pose, then score the z at arrival against `goal`.

    Note the goal z is encoded as if it were step 0 (`is_first=True`, no history)
    while the z at arrival carries the episode's history, so this also prices in
    that context mismatch.
    """
    last = list(posterior_rollout(agent, env, scripted_policy(plan), device, max_steps=len(plan), seed=seed))[-1]
    pose = agent_pose(unwrap_env(env))
    S = goal.shape[1]

    matched = groups_matched(last.stoch, goal)
    resample = agent.rssm.get_dist(last.logit).rsample()
    floor = groups_matched(last.stoch, resample)
    return {
        "plan_len": len(plan),
        "arrived": pose == (target_pos, target_dir),
        "groups": matched,
        "full": matched == S,
        "reward": reward_of(agent, spec, last.stoch, last.logit, goal),
        "floor_groups": floor,
        "floor_full": floor == S,
    }
