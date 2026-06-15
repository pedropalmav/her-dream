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

So `dreamer.py` here is the pre-fork `330ea03` code, differing **only** by the
`.mode()` → `.mode` property fix (see gotcha #2). Any crash in the model code is inherited
from that original world model, **not** introduced by a commit on this branch.

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

2. **`TypeError: 'Tensor' object is not callable` at `...mode()`** inside `_cal_grad`.
   **Root cause: a property called like a method in the original fork.** In
   `distributions/distributions.py`, every distribution defines `mode` (and `mean`) as a
   `@property` returning a `torch.Tensor`. But the pre-fork `dreamer.py` calls it as
   `.mode()` (with parens) on several lines (imagination/replay targets, decoder preds),
   so `dist.mode` yields a Tensor and the trailing `()` tries to **call the Tensor**. Read
   the traceback carets carefully: they span `...().mode` (not the final `()`), i.e. the
   object being called is `.mode`, a Tensor.

   This fails in **eager too** — it is not a `torch.compile` / dynamo / Python-3.12 / frozen-
   module problem (those were earlier wrong guesses). The fork is simply internally
   inconsistent: line 265 already uses `action_dist.mode` and line 453 uses `.mean`
   (no parens, correct), while the imagination/replay/decoder lines use `.mode()` (broken).
   `main` has the correct `.mode` everywhere — the project fixed this after the fork.

   Fix applied: `.mode()` → `.mode` (the only change vs the pre-fork `dreamer.py`).
   Frozen modules and the `r2dreamer` boolean-mask are left exactly as in the original.
   (Optional, unrelated to this crash: `model.compile=false` disables torch.compile, and
   the project is documented/locked for Python 3.11 though it currently runs on 3.12.3.)

## Commands

```bash
# Train the original vanilla WM on Crafter (uv-managed env). $@ are Hydra overrides.
bash run_crafter.sh logdir=./logdir/original_wm_crafter/01 env=crafter seed=1

# With task spooler on a chosen GPU:
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash run_crafter.sh \
  logdir=./logdir/original_wm_crafter/02 env=crafter seed=2 \
  trainer.steps=1010000 trainer.update_log_every=1000 model.rssm.post_use_deter=false

# Direct invocation
python3 train.py logdir=./logdir/test env=crafter

# Disable torch.compile (debugging fallback)
python3 train.py env=crafter model.compile=false

# Representation loss variant (r2dreamer | dreamer | infonce | dreamerpro)
python3 train.py env=crafter model.rep_loss=dreamer

# Ablation: posterior conditioned only on the embedding (no h), default is true
python3 train.py env=crafter model.rssm.post_use_deter=false

# Monitor training
tensorboard --logdir ./logdir
```

### Task spooler (ts) — managing queued runs on the server

```bash
ts                  # list all own jobs
ts -t {process_id}  # tail the current output of a job
ts -c {process_id}  # cat the full output of a job
ts -k {process_id}  # kill a job
```

### Pulling tensorboard logs from the server

```bash
# Copy a run's logdir from barto to the local machine, then view it
scp -r iamonardes@barto.ing.uc.cl:/home/iamonardes/her-dream/logdir/z_without_history_wm_crafter/ \
  ./logdir/z_without_history_wm_crafter/
tensorboard --logdir ./logdir/original_wm_crafter/01
```

`run_crafter.sh` uses `uv run`. Crafter renders to numpy, so no `xvfb` / `MUJOCO_GL` is
needed (those are only for MuJoCo envs).

Note: `main` keeps its run recipes in `execution_commands.md`, but those are for the
goal-conditioned system (`random_goal.sh`, `post_train.sh`, `mission_text`, text-encoder
distillation, etc.) and do **not** apply here — this branch only has `run_crafter.sh`.
Only the operational commands above (ts / scp / tensorboard) carry over.

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
