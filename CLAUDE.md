# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A PyTorch implementation of **R2-Dreamer** (ICLR 2026) extended with goal-conditioned learning where the goal is a discrete stochastic latent variable `z`. The project adds:
- A goal-conditioned actor/critic that receives a one-hot discrete goal concatenated to the RSSM feature vector.
- A `TextEncoderGRU` that maps textual mission descriptions to RSSM-compatible `z` logits, trained via KL divergence against the posterior.
- A family of goal-conditioned reward functions in `rewards.py`, selected by the top-level `goal_type` config.

## Repository layout & installation

The reusable library lives in an **installable package, `her_dream/`** (distribution name `her-dream`, defined in `pyproject.toml`). Install it editable with **`uv pip install -e .`**; this replaces the old `sys.path`/CWD-based imports, so `import her_dream` (and `from her_dream.dreamer import Dreamer`, `from her_dream.envs import make_envs`, …) works from any directory.

- **Package** (`her_dream/`): the core modules `dreamer.py actor_critic.py plan2explore.py rssm.py deter.py rewards.py goals.py trainer.py` and the subpackages `buffers/ distributions/ envs/ networks/ optim/ tools/`, plus the Hydra `configs/` tree (shipped as package data via `[tool.setuptools.package-data]`). Everything here is imported as `her_dream.<module>`; `her_dream/__init__.py` re-exports the common entry points (`Dreamer`, `ActorCritic`, `OnlineTrainer`, `make_envs`/`make_env`, `make_buffer`, `make_reward`, `make_goal_spec`, …).
- **Repo-root scripts** (kept out of the package, but they `import her_dream`): `train.py`, `evaluate.py`, `eval_text_goal.py`, `export.py`, `visualization.py`, and the `experiments/`, `viz/`, `zflow/`, `scripts/`, `tests/` directories.
- **Convention for this doc**: paths naming a library module (e.g. `dreamer.py`, `envs/wrappers.py`, `configs/goal_type/full.yaml`) are relative to `her_dream/` — i.e. `her_dream/dreamer.py`, `her_dream/configs/…`. Root-level things (`train.py`, `experiments/`, `scripts/`, `docs/`) are named as-is.
- Because `configs/` now ships inside the package, the Hydra entry points (`train.py`, `evaluate.py`) resolve it via `pathlib.Path(her_dream.__file__).parent / "configs"` rather than a path relative to the script.

## Commands

**Always run Python / project code through `uv`** (e.g. `uv run python3 train.py ...`, `uv run python3 -c "..."`), not bare `python3`. The commands below are written with bare `python3` for brevity but should be invoked via `uv run`.

```bash
# Install dependencies + the her_dream package, editable (Python 3.11, Ubuntu 24.04)
uv pip install -e .

# The lava env comes from the sibling `cookie-env` package, which `pyproject.toml`
# pins to a git commit. To work against an unreleased cookie-env, override it with
# a local editable install (and remember the pin is what the server resolves):
uv pip install -e ../cookie-env

# Basic training run (env defaults to random_goal, goal_type=full, goal_sample=buffer).
# NOTE the goal_type default is `full`, which does not learn — pass goal_type=row_by_row
# for anything real. Likewise goal_sample=buffer: prefer env.goal_sample=image or
# imagination, since buffer goals are usually unreachable. See "Goal-conditioned rewards" below.
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
python3 train.py env=lava_grid reward_offset=1.0   # terminating env: needs the offset
python3 train.py env=crafter
python3 train.py env=dmc_vision env.task=dmc_walker_walk

# Run on a server with GPU + headless rendering (uses uv); scripts/train.sh is
# the single generic launcher — the mode is selected by the args (load_from=,
# train_text_only=, wm_only=, ...). On the `barto` box always enqueue it instead:
# `CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh ...` — see "Running on barto".
bash scripts/train.sh logdir=./logdir/run/01 seed=1 env.mission_text=True
bash scripts/train.sh logdir=./logdir/posttrain/01 load_from=./logdir/wm_only/01 freeze_wm=True

# Monitor training
tensorboard --logdir ./logdir

# Code formatting
pre-commit run --all-files
```

For headless MuJoCo environments: `export MUJOCO_GL=egl`.

## Running on `barto` (the GPU box)

`barto` has two RTX A6000s and runs **task spooler** (`ts`, `/usr/local/bin/ts`, v2.0.0
with GPU support). **Never launch `scripts/train.sh` directly there — enqueue it, and
use `ts -g` to see who holds a GPU.** The server runs one job at a time (`ts -S` → 1),
so queued runs wait their turn instead of fighting each other for memory; the run
history is 12–20h jobs, long enough that a collision costs a day.

The established form — `execution_commands.md` is full of it, and is the place to look
for whole run recipes:

```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh logdir=./logdir/run/01 seed=1 env=random_goal
```

**`ts -G 1` is the part that does the work.** It reserves a GPU in the queue *and*
assigns one, exporting `CUDA_VISIBLE_DEVICES` into the job itself — which means it
**overrides the prefix**, it does not obey it. Measured: a job enqueued as
`CUDA_VISIBLE_DEVICES=0 ts -G 1 ...` ran with `CUDA_VISIBLE_DEVICES=1` on GPU 1. So
the prefix above is inert whenever `-G` is present; it is kept in the recipes out of
habit, and copying it is harmless but buys nothing.

Where the prefix *does* matter is a run that skips `ts`, because `scripts/train.sh` does
`export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}` — with nothing in front of it,
a direct launch pins **GPU 0** every time, which is how two runs end up on one card
while the other A6000 sits idle. That is a reason to enqueue, not a reason to set the
variable.

To actually choose a card, use ts's own flag: **`ts -g <ids>`** pins those indices
(without checking they are free). Otherwise let `-G 1` pick.

```bash
ts            # the queue, same as `ts -l`: id, state, exit level, runtime, GPUs, command
ts -g         # which jobs hold which GPU ids — the "is a card free?" view
ts -t [id]    # tail -f a job's output (last run if no id); -c for the whole log
ts -o [id]    # path of the job's output file (/tmp/ts-out.*)
ts -i [id]    # job info; -p pid; -s state; -F the full command it was queued with
ts -k [id]    # SIGTERM the job; -r remove a queued one; -w block until it finishes
ts -C         # clear finished jobs off the list
```

**Never use `ts -W` / `ts -D` (job dependencies) on barto.** On this ts 2.0.0 a dependent
job kills the ts server. Reproduced twice on 2026-09-25: the dependent job never shows up
in `ts -l`, and the server dies when the job it depends on finishes. A running job
survives as an orphan (still training, but gone from the queue and from `ts -g`). The
restarted server comes back **empty and with `-S 1`**, so re-run `ts -S 2` after any
restart. To chain runs, enqueue a GPU-less waiter job instead, one that polls for the
first run to finish and then calls `ts -G 1 ...` itself (see `~/lava_posttrain_chain.sh`
on barto).

`-g <ids>` pins specific indices *without* checking they are free, and
`TS_VISIBLE_DEVICES` restricts which GPUs `ts` will hand out at all. A card counts as
available below `ts --get_gpu_free_perc` (90) percent memory used.

`nvidia-smi` still shows raw utilisation, but only `ts -g` says *whose* job owns a
card — check it before assuming one is free.

## Goal-conditioned rewards (`rewards.py`) and goal behavior (`goals.py`)

`make_reward(config)` dispatches on `goal_type` (top-level config key, propagated to env/model/buffer/trainer):

| `goal_type` | Goal shape | Reward |
|---|---|---|
| `first_row` | `(K,)` | 0 if `state[:, 0, :] == goal` exactly, else -1 |
| `row_by_row` | `(S, K)` | `(matching rows / S) - 1` — dense in [-1, 0] |
| `full` | `(S, K)` | 0 if **all S groups** match exactly, else -1 (**config default, but a dead end in practice — see below**) |
| `argmax_full` | `(S, K)` | like `full` but compares against `argmax` one-hot of the prior logits instead of a sample |
| `log_prob` | `(S, K)` | 0 if `dist.log_prob(goal) >= log(prob_threshold)`, else -1 |
| `prob` | `(S, K)` | `dist.log_prob(goal).exp()` — dense in [0, 1] |
| `max_cosine` | `(S, K)` | max-normalized dot product `(g·s)/max(‖g‖,‖s‖)²` between the flattened latents; 1 when equal |

**Do not start new work on `goal_type=full`.** It is the config default for historical
reasons, not because it is the right choice — the exact all-groups match almost never
fires, so the reward is a constant and there is nothing to learn from:

- The match probability is the arithmetic, not bad luck: an exact 32-group
  sample-vs-sample match has **P ≈ 4e-13** under the text encoder's distribution
  (`text_wm_alignment`, 2026-06-02). That is what blocked the text-goal runs.
- On `random_goal` with buffer goals, frozen-WM post-training never leaves the -1001
  floor, while the identical recipe on `fixed_goal` reaches ≈ -430/-500 — so `full`
  only survives where the goal cell never moves.
- On `lava_grid`, a 5000-step run logs `episode/score 0.0` on every episode: the
  reward never fires once.

**Prefer a dense goal type.** `row_by_row` is the one with real mileage here — it is
`(matching rows / S) - 1`, so partial agreement still produces a gradient, and it is
what the runs in the server queue actually use. `prob` and `max_cosine` are the other
continuous options (both already non-negative, so they need no `reward_offset`).

Two honest caveats on `row_by_row`, so it is not oversold: the `row_by_row` +
imagination runs plateau at ≈ -290 while the videos show the agent walking away from
the green square, and `experiments/goal_observation_eval.py` exists precisely to settle
whether that score means the task was done. A dense reward buys a gradient, not a
guarantee that the gradient points at the task.

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

### `reward_offset` — the per-step baseline

Whatever a goal type returns, `make_reward` wraps it in `rewards.with_reward_offset`, which
adds the top-level `reward_offset` (default `0.0`, mirrored into `model`). It is a **run-level
knob, not a goal-type descriptor**: no `configs/goal_type/<type>.yaml` carries one, and at 0
the wrapper returns the reward function itself, so existing runs are bit-identical (dtype
included) and configs saved before the key existed still resolve to 0.

```bash
python3 train.py reward_offset=1.0   # {-1, 0} -> {0, 1}, [-1, 0] -> [0, 1]
```

**Any env that can terminate needs it.** With a negative per-step reward, ending an episode is
worth *more* than surviving it, because termination stops the accumulation of -1s — so an agent
that can perceive termination is rewarded for dying. On a lava variant of the grid env (random
policy, `gamma=0.997`) running to the time limit returns -309 while dying returns -136, and
through the real `_lambda_return` the terminating step has advantage **+90.4** under `{-1, 0}`
versus **-212.1** under `{0, 1}`. `env=lava_grid` is the env this is for — stepping into lava
terminates, so `train.py:validate_config` prints a warning when a `lava-grid` run leaves
`reward_offset` at 0 (a warning, not an error: running lava under `{-1, 0}` is exactly the
measurement quoted above). Everything else here still only truncates, and absent termination the
two schemes are equivalent for the optimal policy — a constant shift moves every value by
`offset/(1-gamma)` and leaves advantages untouched — so this only changes what is learned once
something can terminate.
`prob` and `max_cosine` are already non-negative and need no offset.

The goal source is `env.goal_sample`:

**Avoid `buffer` goals for new work — use `image` or `imagination`.** A state from a past
episode is usually infeasible in the current one: whatever changes between resets (the
green square in `random_goal`, the lava layout in `lava_grid`) is baked into that `z`, so the
episode is handed a goal it cannot reach. It is still the config default, for the same
historical reasons as `goal_type=full`. `image` and `imagination` both build the goal from
the *current* episode, so it is reachable by construction. The evidence so far:
- `random_goal` with buffer goals never leaves -1001 under frozen-WM post-training, while
  the same recipe on `fixed_goal` learns (see "What works" at the end).
- `lava_grid` with buffer goals, 1M steps (`logdir/lava/{rowbyrow_1M/01,02,maxcosine_1M/01,maxcosine_prob_1M/01}`,
  2026-09-25): the death rate stays flat at 5–20% for the entire run, the `row_by_row` score
  drops (≈420 → ≈340, both seeds), and only `max_cosine` on logits rises. That rise is not
  evidence of learning with an unfrozen WM, since logits are not shift-invariant.

- `buffer` — sample a `z` (stoch) from the replay buffer (a state the agent actually visited). Infeasible in most episodes; see above.
- `text` — sample from the `TextEncoderGRU` given the episode's mission (requires `mission_text=True`).
- `random` — random one-hot rows generated by the `GoalConditioned` wrapper.
- `imagination` — at episode start, roll out the (frozen) WM for `goal_imag_horizon` (model config, defaults to `imag_horizon`=15) steps with uniformly random actions from the first observation and take the final prior `z` as the goal, so it is reachable from the current episode by construction (`Dreamer.imagine_goal`).
- `image` — at episode start, the `GoalImageObservation` wrapper (`envs/wrappers.py`) renders the observation of a synthetic state with an auxiliary `FixedGoal` env (`envs.goal_image.GoalImageGenerator`, given an env factory): green square at the current episode's goal position, agent at a random cell. The frozen WM encodes it into `z` with one posterior step from the initial state (`Dreamer.encode_observation`). Designed for `random_goal`, where buffer-sampled goals encode the wrong square position. On `lava_grid` the wrapper copies this episode's lava onto the auxiliary env too (cookie-env's `lava_pos`), so the rendered state is one the live episode can reach; a sampled agent is drawn from the interior minus those cells, since standing on lava is a death state rather than a goal worth handing the agent.

`GoalConditioned` (`envs/wrappers.py`) adds the `goal` key to observations, keyed off the `GoalSpec`: `(K,)` one-hot `MultiBinary` for `scope=first_row`, a float `(S, K)` `Box` when `goal_repr=logit`, else `(S, K)` one-hot `MultiBinary`. Dimensions are enforced via `env.stochastic_classes = model.rssm.discrete` and `env.stochastic_rows = model.rssm.stoch` in the config.

## Training pipelines

A single entry point — **`train.py`** — covers three modes, selected purely from the config (see `train.py:validate_config`/`load_checkpoint`):
- **From-scratch** (default, `load_from=null`) — end-to-end training (WM + actor/critic). With `wm_only=True` it trains only the WM with random actions.
- **Post-training** (`load_from=<logdir>`) — loads a checkpoint, optionally freezes the WM (`freeze_wm=True`), and trains actor/critic on top. Used to isolate "is the WM or the policy the problem".
- **Plan2Explore pretraining** (`plan2explore=True`) — trains the WM, a disagreement ensemble and a goal-agnostic *explorer* actor-critic; the task actor-critic is left at its init. Exclusive with the other modes and takes no `load_from`. Phase 2 is plain post-training on the result.
- **Text distillation** (`train_text_only=True load_from=<logdir> mission_text=True`) — trains only the `TextEncoderGRU` (KL against a frozen WM posterior); WM and actor/critic stay fixed.

There is a single launcher script, `scripts/train.sh` (uv + headless rendering); the mode is chosen entirely by the args it forwards to `train.py` (e.g. `load_from=`, `freeze_wm=True`, `train_text_only=True mission_text=True`).

Diagnostic scripts live in `experiments/` (WM/text-encoder stochasticity, posterior consistency across trajectories, `text_wm_alignment.py` which pairs missions with WM posteriors over real rollouts).

**Layout control in `experiments/`**: a script that measures a WM has to dictate the grid rather than read it off a random reset, which is what `experiments/common.pinned_layout_cfg` builds. For the goal-grid runs it swaps in the pinnable `fixed-goal_` env — `random_goal` and `fixed_goal` render identically and differ only in where the square goes. **A `lava-grid` run keeps its own env**: its WM was trained on observations that contain lava, so measuring it on a lava-free grid would measure it off-distribution, silently and without erroring. The helper therefore *requires* `lava_pos` on a lava run — pinning only the square and the agent leaves the hazards resampling and pins nothing. Two consequences downstream: `goal_observation_eval.py`'s BFS oracle routes around the lava it was given (a route through it would end the episode the oracle is driving), and `goal_position_in_z.py` holds one lava layout for the whole sweep and drops those cells from it, since the square cannot stand on one. `tools/`/`viz/` contain the interactive Crafter dash and trajectory replay utilities.

`success_rate.py` is the comparable benchmark: a frozen suite of 1000 `(initial_state, goal_state)` pairs (`experiments/common/pairs.py`) run against any number of checkpoints, reporting how often the policy reaches the goal pose next to a random-action floor and the BFS-oracle z-check. One suite per env, picked by `--suite` (default: from the run's task) — `random_goal` samples the tasks (random spawn pose, goal cell drawn independently of the green square, ≥4 BFS actions away); `lava_grid` asks a lava run the very same tasks, pair for pair, with the run's `n_lava` cells pinned in each (own seed, never on the spawn/goal/square, never walling the goal off), driven on the run's own lava env and reporting a death rate per driver; and `fixed_goal` *enumerates* them: from the one spawn the agent trained at, every interior cell in every facing (the 252-pose `sweep` block), plus 748 sampled pairs, reported per block since only the sweep is in-distribution. Results land in `experiments/out/success_rate_<suite>.{json,png}`, plus a per-cell success map for the sweep. Shared helpers live in `experiments/common/` (`grid.py` BFS/geometry, `pairs.py` suite, `oracle.py`, plus loading/rollout/metrics/io) — put anything reusable there rather than in a script.

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

**Lava environment** (`env=lava_grid`, suite `lava-grid`): the same room plus `env.n_lava` lava cells, resampled every reset after the goal and the agent so they can never spawn on either. It is **not** a module here — it is `LavaGrid` from the sibling [cookie-env](https://github.com/pedropalmav/cookie-env) package, wired up in `envs/__init__.py`; extend it there, not by cloning a copy into `envs/`. Two things make it different from every other env in this repo:
- **It can terminate.** Stepping into lava sets `terminated`, which reaches the buffer as `is_terminal`. Reaching the green square still does not terminate (inherited from cookie-env's `GoalGrid`). Hence `reward_offset=1.0` — see above.
- **It needs `model.use_cont_head=True`, on the WM run as well as the post-train.** The head is off by default, and without it `Dreamer._imagination_cont` is all ones: no imagined episode ends, so the actor (trained only in imagination) never learns that lava kills. Only the replay critic sees `is_terminal`. Every lava run up to 2026-09-27 had it off. A frozen WM trained without the head cannot supply one, so switching it on means training a new WM. `train.py:validate_config` warns about this.
- **Its own reward is discarded.** `trainer._apply_reward` overwrites `reward` with the goal-conditioned latent reward before the transition is stored, so cookie-env's `lava_penalty` would be inert and is deliberately not forwarded.
- **Its layout has to be pinnable.** Lava resamples on every reset, which breaks anything that has to reproduce a state: the goal image, and the analysis scripts. `env.lava_pos` (cookie-env's `LavaGrid.lava_pos`) pins the cells instead — null in training, set by `GoalImageObservation` and by `experiments/common.pinned_layout_cfg`. `n_lava` only says how many to resample.

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

## What works — configurations that learn, and ones that don't (as of 2026-09)

The full write-up is `docs/experimentos/` on the **`docs/experimentos` branch** (not
merged into `main`): `README.md` has the master table of every run, and one doc per
finding. The numbered run recipes are in that branch's `execution_commands.md`. This
section keeps only the conclusions. Before building on a number, check the doc it
comes from, because most of these are **1 seed per arm**.

**Reading a score:** scores are only comparable at the same `goal_type` **and** the
same goal source, because the score measures reward against *that* goal. On
`random_goal`, `row_by_row` has a floor of -1001, but a random policy already sits
at ≈ -700, so "not learning" means flat around -690. `prob`/`max_cosine` use their
own [0, 1000] scale, where 0 is the floor. The one A/B in the folder that is directly
comparable is `her_strategy` (it changes only the batch relabelling).

### The best recipe so far (`random_goal`)

It has two stages. Every knob in it was an A/B win:

```bash
# 1) WM only, random actions, agent AND square randomised, posterior without h
bash scripts/train.sh logdir=./logdir/random_goal/wm_only_randomstart_no_deter/01 \
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=random env.agent_start_random=True \
    buffer=normal wm_only=True model.rssm.obs_use_deter=False \
    env.steps=1000000 trainer.update_log_every=1000
# 2) post-train on the frozen WM (repeat obs_use_deter=False, or load_state_dict fails)
bash scripts/train.sh load_from=./logdir/random_goal/wm_only_randomstart_no_deter/01 \
    logdir=./logdir/random_goal/posttrain_no_deter_rowbyrow_goalimage_herfuture/01 \
    freeze_wm=True wm_only=False env=random_goal seed=1 mission_text=False \
    env.agent_start_random=True env.goal_sample=image \
    buffer=her buffer.her_strategy=future goal_type=row_by_row \
    model.rssm.obs_use_deter=False env.steps=1000000 trainer.update_log_every=1000
```

This reaches **eval ma25 -126** (train ma25 -91, ≈ 28/32 rows matched). What each
piece is worth:

| knob | good setting | vs | evidence (random_goal, 1M) |
|---|---|---|---|
| goal source | `env.goal_sample=image` (or `imagination`) | `buffer` | buffer goals come from other episodes, so they encode the wrong square cell: -1001. `image` edges out `imagination` (eval -169 vs -217 under `final`), with the caveat that different goal sources are not strictly comparable |
| reward | `goal_type=row_by_row` | `full`, `prob` | `full` stays at -1001 (the 32/32 match is too sparse). `prob` stays flat at 0 on random_goal with or without HER / `h`, and only learns on `fixed_goal` (+331) |
| training regime | frozen-WM post-train (`load_from` + `freeze_wm=True`) | joint end-to-end | joint plateaus at ≈ -306/-309 **even with 2M steps**, vs -180 for post-train. With imagination/image goals the WM generates the goal, so training both chases a moving target |
| posterior | `model.rssm.obs_use_deter=False` | default (with `h`) | -180 vs -290 ma25, 2–4× faster, **in post-train only**. Joint flips it (eval -342 without `h` vs -257 with) |
| HER relabelling | `buffer=her buffer.her_strategy=future` | `final` (config default) | ~3× faster, eval -126 vs -169 (image) and -166 vs -217 (imagination). `final` gives all 65 steps of a sequence the same goal, hundreds of steps away |
| WM pretraining data | `env.agent_start_random=True` | fixed agent start | widens the WM's coverage to the whole grid. This is the WM behind every random_goal run that learns |

### What still isn't proven

**The goal is to reach an arbitrary target `z`, not the green square.** The square is
visual noise, so "the agent ignores the green" (seen in the `imagination` videos) is not
a failure. Some docs in `docs/experimentos/` frame it as one; that framing is wrong.
`image` is just another source of `z` goals. What the score does not prove is that the
agent reaches the *state* the goal `z` describes, as opposed to scoring partial row
matches. Checking that means running `experiments/goal_observation_eval.py` (ORACLE /
POLICY `reach_pos_rate` / CONFUSION) on the best checkpoints, which is still the #1 open
item.

### Dead ends (don't re-run without a new reason)

- `goal_type=full` (see above) and `prob` on `random_goal`.
- `goal_sample=text` / `random`, all -1001. Those runs were on `random_goal` + `full`,
  so the text encoder was never isolated. Redoing them on `fixed_goal` with a
  reachable reward is an open item, not a closed one.
- `goal_sample=buffer` on anything whose layout changes between resets (see the goal
  source list above). It only ever worked on `fixed_goal`.
- `lava_grid` joint end-to-end with buffer goals (the 1M runs of 2026-09-21): death
  rate flat at 5–20%. None of the knobs in the table above were used there.

### Known pitfalls

- **Configs saved before 2026-07-02 with an exact-match reward**
  (`first_row`/`row_by_row`/`full`/`argmax_full`): the reward compared float one-hots
  with `==`, and that broke on GPU (`train/rew` stuck at exactly -1.0, meaning zero
  gradient). The fix (`025f3b3`) compares argmax indices (`rewards._mode_idx`). A
  `train/rew` of exactly -1.0 in an old run is that artifact, not a finding.
- `load_from` rebuilds the model from the **current** config, so architecture
  overrides (`model.rssm.obs_use_deter=False`, RSSM size) have to be repeated on the
  post-train command.
