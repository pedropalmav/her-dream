# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Branch-specific file.** This is the CLAUDE.md for the `feat/original-dreamer-wm`
> branch. It deliberately differs from `main`'s CLAUDE.md because this branch is a
> snapshot of the **pre-fork, vanilla R2-Dreamer world model**, before any
> goal-conditioning / HER / text-encoder work was added. See "Relationship to main".

## What this branch is

A PyTorch implementation of **R2-Dreamer** (ICLR 2026) in its **original, vanilla
form**: a world model + actor-critic that learns from the **environment reward**, with
**no goal-conditioning**. The goal of this branch is to train that original world model
as a clean **baseline on Crafter**.

Concretely, on this branch:
- The actor/critic input is just the RSSM feature vector `feat` (`self.rssm.feat_size`),
  **not** `[feat, goal]`.
- Reward and continuation come from learned **heads** on the world model
  (`self.reward`, `self.cont`), trained against `data["reward"]` / `data["is_terminal"]`.
  There is **no** `reward_function(z, goal)`.
- There is **no** `goal` key, no `GoalConditioned` wrapper involvement, no HER buffer,
  no `TextEncoderGRU`, no `_apply_reward` reward overwrite in the trainer. Those concepts
  do not exist on this branch (grep `goal` in `dreamer.py`/`trainer.py` → nothing).

## Relationship to `main` (read this before debugging)

This branch **forked from `main` at commit `330ea03` (Merge PR #4 `feat/her`,
2026-03-23)** — a very early point. `main` has since advanced to ~PR #26 and is now a
fully goal-conditioned system (actor/critic take `[feat, goal]`, imagined reward comes
from `reward_function(z, goal)`, the WM reward/continue heads were **removed** in commits
`eb21b37` / `fdaec0f`, a `TextEncoderGRU` was added, etc.). `main`'s CLAUDE.md describes
*that* system, which is **not** what runs here.

This branch only adds two commits on top of the fork point, and **neither touches
`dreamer.py` or `trainer.py`**:
- `fdc2e01` — adds `run_crafter.sh` and pulls `pyproject.toml` + `uv.lock` from `main`
  (deps only; torch stays `==2.8.0`).
- `6fd5ee5` — removes the hardcoded `stochastic_classes` line in `train.py` (see below).

So `dreamer.py` here is **byte-identical to the pre-fork `330ea03`**. Any crash in the
model code is inherited from that original world model, **not** introduced by a commit on
this branch.

An alternative strategy to reach the same "vanilla Crafter" goal — adding a
`goal_conditioned` flag on top of `main` and restoring the heads from `10f4afc` — is
described in the plan `~/.claude/plans/quiero-que-crees-una-cozy-rain.md`. This branch
takes the *other* route: fork the original vanilla WM directly.

## Known failure modes (gotchas)

1. **`ConfigKeyError: Key 'stochastic_classes' is not in struct`** (fixed in `6fd5ee5`).
   `train.py` used to do `config.env["stochastic_classes"] = config.model.rssm.discrete`.
   That key only exists in `configs/env/random_goal.yaml` (the original default env), and
   Hydra/OmegaConf configs are in struct mode, so the assignment fails for `env=crafter`.
   This is **not** a regression from any commit — the original code only ran with
   `random_goal`. The line was simply removed (Crafter is vanilla and needs no goal).
   If you ever need to add a key to a struct config, use `with open_dict(config.env): ...`.

2. **`TypeError: 'Tensor' object is not callable` at `self._frozen_reward(imag_feat)`**
   inside `_cal_grad`. **Root cause: `torch.compile` mis-resolves the `deepcopy`'d frozen
   submodules.** `_cal_grad` is wrapped in `torch.compile(mode="reduce-overhead")`. The
   `_frozen_*` modules are created in `clone_and_freeze` via `copy.deepcopy` with shared
   `.data`, and `clone_and_freeze` is called again inside `.to()` (i.e. *after* compile).
   Dynamo represents those frozen attributes as a Tensor, so calling them raises
   "'Tensor' object is not callable". The **live** modules (`self.reward`, etc.),
   registered in `__init__` before compile, trace fine — which is why `self.reward(feat)`
   in the WM loss never fails. This whole frozen-module pattern was **removed in `main`**.

   > Note: an earlier guess blamed a graph break from the boolean-mask indexing
   > `c[off_diag_mask]` in the `r2dreamer` branch. That break was real and was removed
   > (replaced by `c.pow(2).sum() - torch.diagonal(c).pow(2).sum()`), but the crash
   > **persisted without it** — so the graph break was not the cause.

   Fix applied: in the imagination/replay target computations, call the **live** heads
   (`self.reward`/`self.cont`/`self.value`/`self._slow_value`) instead of the `_frozen_*`
   copies. Value- and gradient-equivalent here, since `imag_feat` is already detached, all
   those outputs only feed `.detach()`'d quantities, and frozen params share `.data` with
   the live ones. Fallbacks if anything still misbehaves under compile:
   - `model.compile=false` (slower, but bypasses dynamo entirely).
   - **Use Python 3.11, not 3.12.** The project is documented/locked for 3.11 (`main`'s
     CLAUDE.md; `uv.lock` is cp311), but the training box runs **3.12.3** — a likely
     amplifier of dynamo issues. Note also API drift: this branch uses `.mode()` (method),
     `main` uses `.mode` (property).

## Commands

```bash
# Train the original vanilla WM on Crafter (uv-managed env). $@ are Hydra overrides.
bash run_crafter.sh logdir=./logdir/original_wm_crafter/01 env=crafter seed=1

# With task spooler on a chosen GPU:
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash run_crafter.sh \
  logdir=./logdir/original_wm_crafter/02 env=crafter seed=2 \
  trainer.steps=1010000 trainer.update_log_every=1000

# Direct invocation
python3 train.py logdir=./logdir/test env=crafter

# Disable torch.compile (debugging fallback)
python3 train.py env=crafter model.compile=false

# Representation loss variant (r2dreamer | dreamer | infonce | dreamerpro)
python3 train.py env=crafter model.rep_loss=dreamer

# Monitor training
tensorboard --logdir ./logdir
```

`run_crafter.sh` uses `uv run`. Crafter renders to numpy, so no `xvfb` / `MUJOCO_GL` is
needed (those are only for MuJoCo envs).

## Architecture overview

**Entry point**: `train.py` instantiates `Buffer`, environments (`make_envs`), `Dreamer`,
and `OnlineTrainer`, then calls `trainer.begin(agent)`.

**`dreamer.py` — `Dreamer` class**: Owns all networks and a single **LaProp** optimizer
with Adaptive Gradient Clipping. `_cal_grad()` is the full training step and is wrapped in
`torch.compile(mode="reduce-overhead")` when `config.compile` (default `True`):
1. Encode observations with `MultiEncoder`.
2. RSSM posterior rollout (`observe`) → `post_stoch`, `post_deter`, `post_logit`.
3. KL losses (`dyn` + `rep`).
4. Representation loss selected by `config.rep_loss` (default `r2dreamer`, Barlow-Twins
   style; this is the branch that contains the boolean-mask graph break).
5. **Reward/continue head losses** against env reward / terminals.
6. Imagination rollout, then actor (policy gradient) and critic (TD/λ-return) updates,
   using the **frozen** copies of reward/cont/value for the targets.

**`clone_and_freeze()`**: `deepcopy`'s encoder/rssm/reward/cont/actor/value/slow_value into
`_frozen_*` attributes and re-shares `.data` with the live params so they stay in sync.
Called at init and again inside `.to()`. (This frozen-module pattern is what interacts
badly with `torch.compile` — see gotcha #2.)

**`rssm.py` — `RSSM`**: latent state `(stoch, deter)`. `stoch` is `(B, S, K)` (S groups, K
categories, default 32×16); `deter` is `(B, D)`. `get_feat` → `flatten(stoch) ++ deter`.

**`deter.py` — `Deter` (Block-GRU)**: deterministic transition (prev `stoch`, `deter`,
action → new `deter`), using `BlockLinear`.

**Configs**: Hydra composes `configs/configs.yaml` + `configs/env/<env>.yaml` +
`configs/model/<size>.yaml`. Default model `size12M`; all model hyperparameters live in
`configs/model/_base_.yaml` (`compile: True`, `rep_loss: "r2dreamer"`). The repo default
env is `random_goal`, so **always pass `env=crafter`** for this branch's purpose.

## Tensor shape conventions

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size |
| `T` | Sequence length |
| `S` | Stochastic groups (`rssm.stoch`, default 32) |
| `K` | Categories per group (`rssm.discrete`, default 16) |
| `D` | Deterministic dimension (`rssm.deter`) |
| `F` | Full feature size = `S*K + D` |
| `T_imag` | Imagination horizon (default 15) |

## Key design decisions

- **No goal-conditioning**: this is the original vanilla Dreamer; reward comes from the
  environment via learned heads. (`main` replaced this with goal-conditioning.)
- **No decoder by default**: `r2dreamer` uses Barlow Twins instead of reconstruction; the
  decoder is only instantiated when `rep_loss="dreamer"`.
- **Single optimizer**: all modules share one LaProp optimizer with AGC. Loss scales in
  `_base_.yaml` control relative contributions.
