# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A PyTorch implementation of **R2-Dreamer** (ICLR 2026) extended with goal-conditioned learning where the goal is a discrete stochastic latent variable `z`. The project adds:
- A goal-conditioned actor/critic that receives a one-hot discrete goal concatenated to the RSSM feature vector.
- A `TextEncoderGRU` that maps textual mission descriptions to RSSM-compatible `z` logits, trained via KL divergence against the posterior.
- A custom `reward_function` in `train.py` that computes reward by comparing the first stochastic group of `z` to the goal.

The key constraint: `env.stochastic_classes` is forced equal to `model.rssm.discrete` in `train.py` (line 74), so the goal vector dimension always matches one discrete group of `z`.

## Commands

```bash
# Install dependencies (Python 3.11, Ubuntu 24.04)
pip install -r requirements.txt

# Basic training run
python3 train.py logdir=./logdir/test

# Training with text encoder on RandomGoal environment
python3 train.py logdir=./logdir/run/01 seed=1 env.mission_text=True model.rep_loss=r2dreamer trainer.steps=500000

# Switch representation loss variant (r2dreamer | dreamer | infonce | dreamerpro)
python3 train.py model.rep_loss=dreamer

# Switch environment
python3 train.py env=crafter
python3 train.py env=dmc_vision env.task=dmc_walker_walk

# Run on a server with GPU + headless rendering (uses uv)
bash random_goal.sh logdir=./logdir/run/01 seed=1 env.mission_text=True

# Monitor training
tensorboard --logdir ./logdir

# Code formatting
pre-commit run --all-files
```

For headless MuJoCo environments: `export MUJOCO_GL=egl`.

## Architecture overview

**Entry point**: `train.py` instantiates `Buffer`, environments (`make_envs`), `Dreamer`, and `OnlineTrainer`, then calls `trainer.begin(agent)`.

**`dreamer.py` — `Dreamer` class**: Owns all neural network components and the single optimizer (LaProp). The `_cal_grad()` method is the full training step:
1. Encodes observations with `MultiEncoder`.
2. Runs RSSM posterior rollout (`observe`) to get `post_stoch`, `post_deter`, `post_logit`.
3. Computes KL losses (`dyn` + `rep`) and optionally `text_kl` from `TextEncoderGRU`.
4. Computes the representation loss branch selected by `config.rep_loss`.
5. Runs imagination rollout (`rssm.prior`) for `imag_horizon` steps.
6. Calls `reward_function(imag_stoch, goal)` to get imagined rewards.
7. Updates actor and critic with policy gradient and TD learning.

**`rssm.py` — `RSSM` class**: Manages the latent state `(stoch, deter)`.
- `stoch` shape: `(B, S, K)` — S groups, K categories (e.g. 32×16).
- `deter` shape: `(B, D)`.
- Feature vector (`get_feat`): `flatten(stoch) ++ deter` → shape `(B, S*K+D)`.
- `_obs_net`: posterior logits from `[deter, embed]`.
- `_img_net`: prior logits from `deter` alone (used during imagination).

**`deter.py` — `Deter` (Block-GRU)**: Deterministic transition. Takes previous `stoch`, `deter`, and action; outputs new `deter`. Uses `BlockLinear` for efficiency.

**`networks/text_encoder.py` — `TextEncoderGRU`**: Input `(B, T, L, V)` one-hot characters → GRU over character dimension → head projects to `(B, T, S, K)` logits. Trained so that its distribution matches the RSSM posterior.

**`envs/wrappers.py`**: `GoalConditioned` adds a `goal` key to observations (one-hot, size = `stochastic_classes`). `MissionGridWrapper` adds a `mission` key encoding the text description as `(MAX_LEN=500, VOCAB_SIZE=46)` one-hot.

**Configs**: Hydra composes from `configs/configs.yaml` (top-level), `configs/env/<env>.yaml`, and `configs/model/<size>.yaml`. The default env is `random_goal` and default model is `size12M`. All model hyperparameters live in `configs/model/_base_.yaml`.

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
- **Goal via `z` first group**: `reward_function` compares `state[:, 0, :]` (first of S groups) against the goal one-hot. The goal vector size equals `K` (= `rssm.discrete`), enforced in `train.py`.
- **Single optimizer**: All modules share one LaProp optimizer with Adaptive Gradient Clipping (`agc=0.3`). Loss scales in `_base_.yaml` control relative contributions.
- **`text_kl` loss**: KL is computed as `dists.kl(post_logit.detach(), text_logit)` — the posterior is the target (detached), not the source, so the text encoder chases the world model's representation.
