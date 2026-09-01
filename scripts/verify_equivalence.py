"""Dump a fingerprint of training, so a refactor can be proven to change nothing.

Runs a matrix of configurations on CPU (deterministic, unlike a GPU run) and for
each one records:
  * every scalar metric from STEPS real `update()` calls,
  * a checksum of every optimizer parameter afterwards, keyed by its
    `_named_params` name, and
  * the output of the two inference paths, `act` and `imagine_goal`.

The parameter checksums are the strong claim: if they all match after STEPS
optimizer steps, the whole training step — world model, actor, critic, lambda
returns, loss scaling, gradient clipping, optimizer — is identical.

Usage
-----
    git checkout <before>
    uv run python3 scripts/verify_equivalence.py /tmp/before.json
    git checkout <after>
    uv run python3 scripts/verify_equivalence.py /tmp/after.json
    diff /tmp/before.json /tmp/after.json      # must be empty

Two traps this script exists to avoid, both of which silently hollow out the
comparison (see STEPS and VariedBuffer below): a GradScaler that has not settled
yet, so no optimizer step ever applies, and a NaN loss, which makes the scaler
skip every step. Both leave the parameters at their init and make any refactor
look equivalent.
"""

import contextlib
import io
import json
import sys
import traceback

import torch

sys.path.insert(0, "/home/pedropalmav/her-dream")

from tests.dreamer.conftest import StubReplayBuffer, make_default_obs, make_real_dreamer  # noqa: E402


class VariedBuffer(StubReplayBuffer):
    """StubReplayBuffer with a varying `direction`.

    The stub emits a constant-zero direction, so the MLP half of the embedding
    is identical across the batch -> std 0 -> the r2dreamer Barlow loss divides
    by (0 + 1e-8), which underflows to 0 in fp16 and yields NaN. That NaN makes
    GradScaler skip every optimizer step, which would silently hollow out this
    whole comparison. Real rollouts have a varying direction.
    """

    def sample(self):
        data, index, initial = super().sample()
        idx = torch.randint(0, 4, (self.B, self.T))
        data["direction"] = torch.nn.functional.one_hot(idx, 4).float()
        return data, index, initial


# GradScaler starts at 65536 and halves on every fp16 overflow; on this tiny CPU
# model it takes ~16 steps to settle at 1.0, and only then does the optimizer
# actually apply updates. Fewer steps than that would compare untouched weights.
STEPS = 25

# (label, kwargs to make_real_dreamer, buffer kwargs)
CASES = []

# Every goal type — covers first_row's (K,) goal, log_prob's threshold one-hot,
# and the logit/prob state representations.
for gt in ("full", "first_row", "row_by_row", "argmax_full", "log_prob", "prob", "max_cosine"):
    CASES.append((f"goal_type={gt}", dict(goal_type=gt), {}))
CASES.append((
    "goal_type=max_cosine,prob",
    dict(goal_type="max_cosine", state_repr="prob", goal_repr="prob"),
    {},
))

# Every action space — each selects a different actor dist.
for act in ("discrete", "multi", "cont", "n"):
    CASES.append((f"act={act}", dict(act=act), {}))

# Every representation-loss branch.
for rl in ("r2dreamer", "dreamer", "infonce", "dreamerpro"):
    CASES.append((f"rep_loss={rl}", dict(model__rep_loss=rl), {}))

# Every training mode.
CASES.append(("mode=wm_only", dict(wm_only=True), {}))
CASES.append(("mode=freeze_wm", dict(freeze_wm=True), {}))
CASES.append(("mode=mission_text", dict(mission_text=True), dict(mission=True)))
CASES.append((
    "mode=train_text_only",
    dict(train_text_only=True, mission_text=True),
    dict(mission=True),
))

# The reward/continue heads from step 1, including the head reward source.
CASES.append(("heads=both", dict(model__use_reward_head=True, model__use_cont_head=True), {}))
CASES.append((
    "heads=reward_source",
    dict(model__use_reward_head=True, model__imag_reward_source="head"),
    {},
))
CASES.append((
    "heads=wm_only",
    dict(wm_only=True, model__use_reward_head=True, model__use_cont_head=True),
    {},
))


def scalarize(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return round(float(value.detach()), 6)
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return None


def run_case(kwargs, buf_kwargs):
    torch.manual_seed(0)
    agent, goal_shape = make_real_dreamer(**kwargs)
    buf = VariedBuffer(goal_shape, **buf_kwargs)

    rows = []
    for _ in range(STEPS):
        torch.manual_seed(1234)
        metrics = agent.update(buf)
        rows.append({k: s for k, v in sorted(metrics.items()) if (s := scalarize(v)) is not None})

    # The decisive check: every optimizer parameter after STEPS updates.
    params = {name: round(float(p.detach().double().sum()), 6) for name, p in sorted(agent._named_params.items())}

    # Inference paths that build the policy input by hand today.
    torch.manual_seed(7)
    obs = make_default_obs(B=2, goal_shape=goal_shape)
    state = agent.get_initial_state(2)
    torch.manual_seed(7)
    action, _, _ = agent.act(obs, state, eval=True)
    act_out = round(float(action.detach().double().sum()), 6)

    torch.manual_seed(11)
    goal = torch.zeros(2, *goal_shape)
    goal[..., 0] = 1.0
    imagined = agent.imagine_goal({
        "is_first": torch.ones(2, dtype=torch.bool),
        **{k: v for k, v in make_default_obs(B=2, goal_shape=goal_shape).items() if k != "is_first"},
    })
    imag_out = round(float(imagined.detach().double().sum()), 6)

    return {"metrics": rows, "params": params, "act": act_out, "imagine_goal": imag_out}


out = {}
for label, kwargs, buf_kwargs in CASES:
    try:
        # Dreamer prints a parameter table on construction; keep it out of the dump.
        with contextlib.redirect_stdout(io.StringIO()):
            out[label] = run_case(kwargs, buf_kwargs)
    except Exception:
        out[label] = {"ERROR": traceback.format_exc().splitlines()[-1]}

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print(f"{len(out)} cases -> {sys.argv[1]}")
