# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A PyTorch implementation of **R2-Dreamer** (ICLR 2026) extended with goal-conditioned learning where the goal is a discrete stochastic latent variable `z`. The project adds:
- A goal-conditioned actor/critic that receives a one-hot discrete goal concatenated to the RSSM feature vector.
- A `TextEncoderGRU` that maps textual mission descriptions to RSSM-compatible `z` logits, trained via KL divergence against the posterior.
- A family of goal-conditioned reward functions in `rewards.py`, selected by the top-level `goal_type` config.

## Repository layout & installation

The reusable library lives in an **installable package, `her_dream/`** (distribution name `her-dream`, defined in `pyproject.toml`). Install it editable with **`uv pip install -e .`**; this replaces the old `sys.path`/CWD-based imports, so `import her_dream` (and `from her_dream.dreamer import Dreamer`, `from her_dream.envs import make_envs`, …) works from any directory.

- **Package** (`her_dream/`): the core modules `dreamer.py actor_critic.py plan2explore.py temporal_distance.py rssm.py deter.py rewards.py goals.py trainer.py` and the subpackages `buffers/ distributions/ envs/ networks/ optim/ tools/`, plus the Hydra `configs/` tree (shipped as package data via `[tool.setuptools.package-data]`). Everything here is imported as `her_dream.<module>`; `her_dream/__init__.py` re-exports the common entry points (`Dreamer`, `ActorCritic`, `OnlineTrainer`, `make_envs`/`make_env`, `make_buffer`, `make_reward`, `make_goal_spec`, …).
- **Repo-root scripts** (kept out of the package, but they `import her_dream`): `train.py`, `evaluate.py`, `eval_text_goal.py`, `export.py`, `visualization.py`, and the `experiments/`, `viz/`, `zflow/`, `scripts/`, `tests/` directories.
- **Convention for this doc**: paths naming a library module (e.g. `dreamer.py`, `envs/wrappers.py`, `configs/goal_type/full.yaml`) are relative to `her_dream/` — i.e. `her_dream/dreamer.py`, `her_dream/configs/…`. Root-level things (`train.py`, `experiments/`, `scripts/`, `docs/`) are named as-is.
- Because `configs/` now ships inside the package, the Hydra entry points (`train.py`, `evaluate.py`) resolve it via `pathlib.Path(her_dream.__file__).parent / "configs"` rather than a path relative to the script.

## Commands

**Always run Python / project code through `uv`** (e.g. `uv run python3 train.py ...`, `uv run python3 -c "..."`), not bare `python3`. The commands below are written with bare `python3` for brevity but should be invoked via `uv run`.

```bash
# Install dependencies + the her_dream package, editable (Python 3.11, Ubuntu 24.04)
uv pip install -e .

# Basic training run (env defaults to random_goal, goal_type=full, goal_sample=buffer)
python3 train.py logdir=./logdir/test

# Training with text encoder on RandomGoal environment
python3 train.py logdir=./logdir/run/01 seed=1 env.mission_text=True model.rep_loss=r2dreamer trainer.steps=500000

# World-model-only pretraining (no actor/critic learning, random actions)
python3 train.py logdir=./logdir/wm_only/01 wm_only=True

# Post-training: load a pretrained WM, freeze it, train actor/critic on top
python3 train.py logdir=./logdir/posttrain/01 load_from=./logdir/wm_only/01 freeze_wm=True

# Plan2Explore: pretrain the WM with an exploration policy, then post-train the
# goal achiever on the frozen result (the second command is the ordinary
# post-training path — the checkpoint's explorer/ensemble are dropped on load).
python3 train.py logdir=./logdir/p2e/01 plan2explore=True
python3 train.py logdir=./logdir/p2e/achiever load_from=./logdir/p2e/01 freeze_wm=True

# LEXA: explorer and achiever trained together in one run, the achiever driven
# by the learned temporal distance rather than an exact z match.
python3 train.py logdir=./logdir/lexa/01 lexa=True model.imag_reward_source=temporal

# Distill the text encoder against a frozen WM
python3 train.py logdir=./logdir/distill/01 load_from=./logdir/wm_only/01 train_text_only=True mission_text=True

# Switch reward variant (see "Goal-conditioned rewards" below)
python3 train.py goal_type=first_row

# Switch goal source (buffer | text | random)
python3 train.py env.goal_sample=text mission_text=True

# Switch representation loss variant (r2dreamer | dreamer | infonce | dreamerpro)
python3 train.py model.rep_loss=dreamer

# Switch environment
python3 train.py env=fixed_goal
python3 train.py env=crafter
python3 train.py env=dmc_vision env.task=dmc_walker_walk

# Run on a server with GPU + headless rendering (uses uv); scripts/train.sh is
# the single generic launcher — the mode is selected by the args (load_from=,
# train_text_only=, wm_only=, ...).
bash scripts/train.sh logdir=./logdir/run/01 seed=1 env.mission_text=True
bash scripts/train.sh logdir=./logdir/posttrain/01 load_from=./logdir/wm_only/01 freeze_wm=True

# Monitor training
tensorboard --logdir ./logdir

# Code formatting
pre-commit run --all-files
```

For headless MuJoCo environments: `export MUJOCO_GL=egl`.

## Goal-conditioned rewards (`rewards.py`) and goal behavior (`goals.py`)

`make_reward(config)` dispatches on `goal_type` (top-level config key, propagated to env/model/buffer/trainer):

| `goal_type` | Goal shape | Reward |
|---|---|---|
| `first_row` | `(K,)` | 0 if `state[:, 0, :] == goal` exactly, else -1 |
| `row_by_row` | `(S, K)` | `(matching rows / S) - 1` — dense in [-1, 0] |
| `full` | `(S, K)` | 0 if **all S groups** match exactly, else -1 (**default; current research focus is this z-full comparison**) |
| `argmax_full` | `(S, K)` | like `full` but compares against `argmax` one-hot of the prior logits instead of a sample |
| `log_prob` | `(S, K)` | 0 if `dist.log_prob(goal) >= log(prob_threshold)`, else -1 |
| `prob` | `(S, K)` | `dist.log_prob(goal).exp()` — dense in [0, 1] |
| `max_cosine` | `(S, K)` | max-normalized dot product `(g·s)/max(‖g‖,‖s‖)²` between the flattened latents; 1 when equal |

Everything else a goal type varies is captured by a **`GoalSpec`** (`goals.py`), built from
three descriptor keys carried in each `configs/goal_type/<type>.yaml` (never string-branch on
`goal_type` — read the spec via `goals.make_goal_spec(config)` and the helpers):

| descriptor | values | meaning / helper |
|---|---|---|
| `state_repr` | `stoch` \| `dist` \| `logit` \| `prob` | reward fn's state arg (`goals.reward_state`); also whether `act()` stashes `logit` (`goals.stashes_logit`) |
| `goal_repr` | `sample` \| `argmax` \| `logit` \| `prob` | the goal tensor, must match the reward comparison (`goals.goal_from_latent`); also env goal space (`MultiBinary` vs float `Box`) + generation (one-hot vs `randn`) |
| `scope` | `first_row` \| `full` | actor/critic goal-input size: `K` vs `S*K` (`goals.goal_size`) |

(`uses_threshold`, `log_prob`-only, appends a threshold one-hot to the actor/critic input.)

`prob` is `softmax(logit)` — the per-group categorical probabilities. It exists for
`max_cosine`: a logit vector is only defined up to a per-group additive constant, which
softmax ignores but a dot product does not, so comparing probabilities makes the reward
shift-invariant and bounded in [0, 1]. It must be set on **both** axes (`make_goal_spec`
rejects a `prob`/`logit` mix), which is also the CLI switch between the two variants:

```bash
python3 train.py goal_type=max_cosine                                  # raw logits (file default)
python3 train.py goal_type=max_cosine state_repr=prob goal_repr=prob   # softmax
```
Adding a goal type = a reward fn in `rewards.py` + a `configs/goal_type/<type>.yaml` with these
keys; no edits to dreamer/trainer/her_buffer/wrappers. `goals.default_descriptors(goal_type)`
reads a type's descriptors from its config file (used to backfill old checkpoints in zflow/viz).

The goal source is `env.goal_sample`:
- `buffer` — sample a `z` (stoch) from the replay buffer (a state the agent actually visited).
- `text` — sample from the `TextEncoderGRU` given the episode's mission (requires `mission_text=True`).
- `random` — random one-hot rows generated by the `GoalConditioned` wrapper.
- `imagination` — at episode start, roll out the (frozen) WM for `goal_imag_horizon` (model config, defaults to `imag_horizon`=15) steps with uniformly random actions from the first observation and take the final prior `z` as the goal, so it is reachable from the current episode by construction (`Dreamer.imagine_goal`).
- `image` — at episode start, the `GoalImageObservation` wrapper (`envs/wrappers.py`) renders the observation of a synthetic state with an auxiliary `FixedGoal` env (`envs.goal_image.GoalImageGenerator`, given an env factory): green square at the current episode's goal position, agent at a random cell. The frozen WM encodes it into `z` with one posterior step from the initial state (`Dreamer.encode_observation`). Designed for `random_goal`, where buffer-sampled goals encode the wrong square position.

`GoalConditioned` (`envs/wrappers.py`) adds the `goal` key to observations, keyed off the `GoalSpec`: `(K,)` one-hot `MultiBinary` for `scope=first_row`, a float `(S, K)` `Box` when `goal_repr=logit`, else `(S, K)` one-hot `MultiBinary`. Dimensions are enforced via `env.stochastic_classes = model.rssm.discrete` and `env.stochastic_rows = model.rssm.stoch` in the config.

## Training pipelines

A single entry point — **`train.py`** — covers three modes, selected purely from the config (see `train.py:validate_config`/`load_checkpoint`):
- **From-scratch** (default, `load_from=null`) — end-to-end training (WM + actor/critic). With `wm_only=True` it trains only the WM with random actions.
- **Post-training** (`load_from=<logdir>`) — loads a checkpoint, optionally freezes the WM (`freeze_wm=True`), and trains actor/critic on top. Used to isolate "is the WM or the policy the problem".
- **Plan2Explore pretraining** (`plan2explore=True`) — trains the WM, a disagreement ensemble and a goal-agnostic *explorer* actor-critic; the task actor-critic is left at its init. Exclusive with the other modes and takes no `load_from`. Phase 2 is plain post-training on the result.
- **LEXA** (`lexa=True`, normally with `model.imag_reward_source=temporal`) — the same explorer and ensemble, but the achiever is trained *alongside* it in imagination instead of in a later run, and data collection alternates: the explorer drives every `model.explore_every_ep`-th episode (default 2), the achiever the rest. The achiever gets no `repval` loss, since that one is fed the analytic goal reward rather than the objective it is being trained on.
- **Text distillation** (`train_text_only=True load_from=<logdir> mission_text=True`) — trains only the `TextEncoderGRU` (KL against a frozen WM posterior); WM and actor/critic stay fixed.

There is a single launcher script, `scripts/train.sh` (uv + headless rendering); the mode is chosen entirely by the args it forwards to `train.py` (e.g. `load_from=`, `freeze_wm=True`, `train_text_only=True mission_text=True`).

Diagnostic scripts live in `experiments/` (WM/text-encoder stochasticity, posterior consistency across trajectories, `text_wm_alignment.py` which pairs missions with WM posteriors over real rollouts). `tools/`/`viz/` contain the interactive Crafter dash and trajectory replay utilities.

## Architecture overview

**`dreamer.py` — `Dreamer` class**: Owns all neural network components and the single optimizer (LaProp). The `_cal_grad()` method is the full training step:
1. Encodes observations with `MultiEncoder`.
2. Runs RSSM posterior rollout (`observe`) to get `post_stoch`, `post_deter`, `post_logit`.
3. Computes KL losses (`dyn` + `rep`) and optionally `text_kl` from `TextEncoderGRU`.
4. Computes the representation loss branch selected by `config.rep_loss`.
5. Runs imagination rollout (`rssm.prior`) for `imag_horizon` steps.
6. Calls `reward_function(imag_stoch, goal)` to get imagined rewards.
7. Hands the rollout to `self.ac` (an `ActorCritic`) for the policy-gradient and TD losses.

`Dreamer` keeps the single optimizer and the single `backward()`: every sub-module
returns unscaled losses which `_cal_grad` weights by `loss_scales` and sums.

**`actor_critic.py` — `ActorCritic` class**: the goal-conditioned actor, its
critic, the slow value target, the return EMA and the three frozen inference
clones (`_frozen_actor` / `_frozen_value` / `_frozen_slow_value`). It owns:
- `policy_input(feat, goal)` — the single definition of the policy input layout,
  `feat ++ flatten(goal) [++ threshold_onehot]`. Everything that runs the policy
  (`Dreamer.act`, `Dreamer._imagine`, both loss branches,
  `experiments/common/rollout.actor_policy`) goes through it.
- `imagination_loss(...) -> (losses, metrics, ret)` — lambda returns, advantage,
  the `policy` and `value` losses.
- `replay_value_loss(...) -> (losses, metrics)` — the `repval` loss, deliberately
  fed the *attached* RSSM feature so the critic's gradient reaches the world model.
- `clone_and_freeze` / `update_slow_target` / `freeze` / `train`.

It is instantiable more than once. A `name=""` constructor argument prefixes every
loss key, metric name and optimizer module name, so an exploration actor-critic
(`name="explore"` → `explore_policy`, `explore_value`, …) can coexist with the task
one; `loss_scales(base)` mirrors the base scales onto the prefixed keys, and
`Dreamer._imagine(start, horizon, goal, actor_critic)` takes the policy to roll out
as a parameter. The default unnamed instance leaves every key exactly as it was.

**Checkpoint layout**: `Dreamer.state_dict()` nests these under `ac.` (`ac.actor.*`,
`ac.value.*`, `ac._slow_value.*`, `ac.return_ema.*`, `ac.threshold_onehot`).
Checkpoints written before that are still loadable — every load site runs
`tools.migrate_agent_state_dict(sd)` first, which rewrites the old flat keys and is
a no-op on current ones.

**`plan2explore.py` — `Disagreement` class**: an ensemble of `disag.models`
one-step predictors, each mapping `feat ++ action` to the next latent
(`disag.target`: `stoch` | `deter` | `feat`). Their spread is the intrinsic
reward:
- `loss(feat, action, target)` — one-step prediction error, summed over members.
  Inputs *and* targets are detached, so the ensemble never shapes the
  representation (as in DreamerV2). Because the buffer already shifts the action
  one step back (`buffers/buffer.py`), the transition `t -> t+offset` pairs
  `feat[:, :-offset]` with `action[:, offset:]`.
- `intrinsic_reward(feat, action)` — std across members, mean over the predicted
  dimensions, times `disag.intr_scale`. Accumulated with Welford rather than
  stacking every member: memory stays flat in the ensemble size, and a well-fit
  ensemble reaches exactly zero instead of the ~1e-4 float-error floor the
  sum-of-squares form leaves behind.

`disag_raw` is logged unscaled — set `disag.intr_scale` against it, because
`ReturnEMA` clips its scale at `min=1.0` and so cannot amplify a tiny reward.
`disag.log=True` takes the log instead, if the raw magnitude is far from O(1).

Under `plan2explore=True`, `Dreamer` builds `self.explorer` (an `ActorCritic`
with `goal_size=0` and `name="explore"`) and `self.disag`; `act` collects data
with the explorer, and only the WM, the ensemble and the explorer are optimized.
Checkpoints therefore carry `explorer.*` / `disag.*`, which `train.py` drops on
load for an agent that does not build them (`_OPTIONAL_MODULE_PREFIXES`).

**`temporal_distance.py` — `TemporalDistance` class**: LEXA's goal-reaching
reward. An MLP over `[flat(stoch), flat(goal)]` regressing the normalised number
of steps between the two, selected with `imag_reward_source=temporal` (an axis
independent of `lexa` — it works in an ordinary run too).
- `loss(stoch)` — pairs drawn from the same imagined trajectory, labelled
  `(j - i) / (T - 1)`, plus `neg_sampling_factor` extra cross-trajectory pairs
  labelled maximally distant so the predictor cannot collapse to a constant.
  Latents are detached, so it never shapes the representation.
- `distance(stoch, goal)` — the reward is its negation.

Why it exists: with `goal_type=full` the analytic reward needs an exact 32-group
`z` match, which is ~never satisfied (P ≈ 4e-13 under the text encoder — see the
investigation notes below). A learned distance gives the achiever a gradient
everywhere instead of an all-or-nothing signal.

Note the predictor sits slightly above its positive-pair labels in practice: the
negatives carry equal loss weight while being a tenth of the samples (LEXA sums
the two means). That is a constant-ish bias, which `ReturnEMA` absorbs;
`temporal_distance.neg_sampling_factor` is the knob.

**`rssm.py` — `RSSM` class**: Manages the latent state `(stoch, deter)`.
- `stoch` shape: `(B, S, K)` — S groups, K categories (e.g. 32×16).
- `deter` shape: `(B, D)`.
- Feature vector (`get_feat`): `flatten(stoch) ++ deter` → shape `(B, S*K+D)`.
- `_obs_net`: posterior logits from `[deter, embed]`.
- `_img_net`: prior logits from `deter` alone (used during imagination).

**`deter.py` — `Deter` (Block-GRU)**: Deterministic transition. Takes previous `stoch`, `deter`, and action; outputs new `deter`. Uses `BlockLinear` for efficiency.

**`networks/text_encoder.py` — `TextEncoderGRU`**: Input `(B, T, L, V)` one-hot characters → GRU over character dimension → head projects to `(B, T, S, K)` logits. Trained so that its distribution matches the RSSM posterior.

**`envs/wrappers.py`**: `GoalConditioned` adds the `goal` key (see above). `MissionGridWrapper` adds a `mission` key encoding the text description (MAX_LEN=500, VOCAB_SIZE=46; stored as ints, one-hotted in the forward pass for memory).

**Goal-grid environments** (`envs/fixed_goal.py`, `envs/random_goal.py`): minigrid-style rooms where the agent must reach a green square. They are identical except that in `fixed_goal` the green square stays at a fixed position (`env.goal_pos_x/y`) across episodes, while in `random_goal` it moves to a random cell every episode. Score -1001 is the floor (goal never reached within the 1000-step time limit); learning shows as scores rising above ~-500.

**Configs**: Hydra composes from `configs/configs.yaml` (top-level), `configs/env/<env>.yaml`, `configs/model/<size>.yaml`, `configs/buffer/<type>.yaml` (`her` with her_ratio=0.8/strategy=final, or `normal`), and `configs/goal_type/<type>.yaml`. The default env is `random_goal` and default model is `size12M`. The `goal_type` group is a `# @package _global_` preset per goal type (selected with `goal_type=<type>`); each file sets the root `goal_type` scalar plus the `state_repr`/`goal_repr`/`scope`/`uses_threshold` descriptors (see the `GoalSpec` table above), which `configs.yaml` fans out to `env`/`model`/`buffer`/`trainer` like `goal_type`. `log_prob.yaml` also carries `prob_threshold`/`prob_threshold_step` (the only goal type that uses them — the model block reads them via `${oc.select:...}` so other types resolve without them). All model hyperparameters live in `configs/model/_base_.yaml`. Every run saves its composed config to `<logdir>/.hydra/config.yaml` — always check that, not the current defaults, when analyzing an old run.

## Tensor shape conventions

Shape annotations throughout the code use these symbols (see `docs/tensor_shapes.md`):

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

- **No decoder by default**: `r2dreamer` uses Barlow Twins instead of reconstruction, avoiding the computational cost of a decoder. The decoder is only instantiated when `rep_loss="dreamer"`.
- **Single optimizer**: All modules share one LaProp optimizer with Adaptive Gradient Clipping (`agc=0.3`). Loss scales in `_base_.yaml` control relative contributions.
- **`text_kl` loss**: KL is computed as `dists.kl(post_logit.detach(), text_logit)` — the posterior is the target (detached), not the source, so the text encoder chases the world model's representation.
- **Frozen-WM post-training** decouples representation learning from policy learning, so failures can be attributed to one or the other.

## Current investigation (2026-06)

With the identical recipe (frozen pretrained WM + `goal_type=full` + `goal_sample=buffer`), post-training **works on `fixed_goal`** (`logdir/post_train_wm_only/{her,normal}_buffer_goals` reach scores ≈ -430/-500) but **fails on `random_goal`** (`logdir/random_goal/posttrain_frozenwm_{herbuf,normalbuf}_goalbuf` stay at -1001). Working hypothesis (branch `analysis/random-goal-z-position`): some groups of `z` encode the green square's position; in `random_goal`, buffer-sampled goals come from episodes where the square was elsewhere, so a full-z match is unreachable in the current episode by construction. Earlier finding (2026-06-02, `text_wm_alignment`): `goal_type=full` is also what blocked the text-goal runs — exact 32-group sample-vs-sample match has P ≈ 4e-13 under the text encoder's distribution. See `bitacora_nano/general.md` for the full timeline.
