# HER-Dream: Goal-Conditioned World Model RL con Objetivos Estocásticos Discretos

## ¿En qué consiste el proyecto?

Este proyecto extiende **DreamerV3** — un algoritmo de reinforcement learning basado en *world models* — para soportar **aprendizaje orientado a objetivos** donde el objetivo es una variable latente discreta y estocástica de tipo `z`.

La idea central es que el agente no solo aprende a modelar la dinámica del mundo, sino que también aprende a *interpretar* un objetivo (ya sea especificado por vector discreto o por descripción textual) y a perseguirlo dentro del espacio latente del world model.

El acrónimo **HER** hace referencia a *Hindsight Experience Replay*, técnica clásica de goal-conditioned RL en la que el agente reutiliza experiencias pasadas imaginando que su objetivo era el estado que realmente alcanzó.

---

## Contexto: DreamerV3

DreamerV3 aprende tres cosas simultáneamente:

1. **World Model**: Un modelo que predice cómo evoluciona el mundo en un espacio latente compacto.
2. **Actor**: Una política que elige acciones para maximizar retornos.
3. **Critic**: Una función de valor que estima retornos esperados.

El estado latente se compone de dos partes:
- `deter`: Estado determinista, calculado por un RNN (Block-GRU).
- `stoch (z)`: Estado estocástico **discreto**, una distribución categórica de forma `(S, K)` con S grupos de K categorías (e.g., 32 × 16 = 512 dimensiones).

La variable `z` es el corazón del proyecto: actúa como representación simbólica del estado del mundo y es la que se usa como "objetivo" del agente.

---

## Contribución principal: Objetivos de tipo `z`

La novedad del proyecto es usar directamente el estado estocástico discreto `z` del RSSM como objetivo del agente.

### 1. Objetivo discreto (vector one-hot)

El entorno `RandomGoal` genera un objetivo aleatorio como vector one-hot de tamaño `stochastic_classes` (e.g., 4 clases). Este objetivo se concatena con el feature vector `[deter, flatten(z)]` antes de pasarlo al actor y al critic, condicionando la política al objetivo deseado.

```
feat = [flatten(z), deter]      # Feature del world model
policy_input = [feat, goal]     # Concatenación con objetivo
```

### 2. Objetivo textual (misión en lenguaje natural)

Cuando `mission_text=True`, se activa un **TextEncoderGRU** que convierte una descripción textual de la misión en logits sobre el espacio de `z`, compatibles con el RSSM:

```
misión (texto) → TextEncoderGRU → logits_z ~ (B, T, S, K)
```

La red aprende a predecir qué estado latente `z` corresponde a una misión dada, usando como supervisión los logits posteriores del RSSM (entrenamiento con KL divergence). Esto permite al agente interpretar instrucciones en lenguaje natural.

### 3. Variantes de loss de representación

El proyecto implementa cuatro variantes del loss de representación del world model, seleccionables con el parámetro `rep_loss`:

| Variante | Descripción |
|----------|-------------|
| `r2dreamer` | Barlow Twins sin decoder (redundancia-reducida) — variante principal |
| `dreamer` | Reconstrucción estándar de observaciones con decoder |
| `infonce` | Contrastive learning (InfoNCE) |
| `dreamerpro` | Prototype matching con SwAV |

---

## Estructura general del proyecto

```
her-dream/
├── train.py                  # Entry point: inicializa entorno, buffer y agente
├── dreamer.py                # Módulo central: world model, losses, actor-critic
├── rssm.py                   # Recurrent State Space Model (posterior/prior)
├── deter.py                  # Block-GRU para transición determinista
├── trainer.py                # Loop de entrenamiento online
├── buffer.py                 # Replay buffer (basado en TorchRL)
├── visualization.py          # Visualización de trayectorias y métricas
│
├── networks/                 # Módulos de red neuronal
│   ├── conv_encoder.py       # Encoder CNN para observaciones visuales
│   ├── conv_decoder.py       # Decoder CNN (solo para rep_loss="dreamer")
│   ├── multi_encoder.py      # Encoder multimodal (imagen + estado bajo)
│   ├── multi_decoder.py      # Decoder multimodal
│   ├── mlp.py                # MLP genérico
│   ├── mlp_head.py           # Cabezas de política y valor
│   ├── projector.py          # Proyector para Barlow Twins / InfoNCE
│   ├── text_encoder.py       # TextEncoderGRU: misión textual → logits z
│   ├── block_linear.py       # Linear por bloques (eficiencia en Block-GRU)
│   ├── lambda_layer.py       # Capa Lambda utilitaria
│   ├── return_ema.py         # EMA para normalización de retornos
│   └── rmsnorm_2d.py         # RMSNorm 2D
│
├── distributions/            # Distribuciones probabilísticas
│   ├── distributions.py      # OneHotDist, MultiOneHotDist, TwoHot, MSEDist
│   ├── functional.py         # symlog / symexp (estabilización numérica)
│   └── constructors.py       # Fábricas de distribuciones
│
├── envs/                     # Entornos de RL
│   ├── random_goal.py        # Entorno principal: MiniGrid con objetivo aleatorio
│   ├── wrappers.py           # GoalConditioned, MissionGridWrapper
│   ├── parallel.py           # Ejecución paralela de entornos
│   ├── crafter.py            # Entorno Crafter
│   ├── dmc.py                # DeepMind Control Suite
│   ├── atari.py              # Atari 100k
│   ├── memorymaze.py         # MemoryMaze
│   └── metaworld.py          # MetaWorld
│
├── optim/                    # Optimizadores
│   ├── laprop.py             # LaProp (adaptativo, estable para world models)
│   └── agc.py                # Adaptive Gradient Clipping
│
├── tools/                    # Utilidades generales
│   ├── logging.py            # Logging a TensorBoard
│   ├── checkpoint.py         # Guardado/carga de checkpoints
│   ├── training.py           # Helpers de entrenamiento
│   ├── math_utils.py         # Funciones matemáticas auxiliares
│   ├── nn_utils.py           # Utilidades de redes neuronales
│   └── torch_utils.py        # Utilidades de PyTorch
│
├── configs/                  # Configuraciones Hydra
│   ├── configs.yaml          # Config principal (valores por defecto)
│   ├── env/                  # Configs por entorno
│   │   ├── random_goal.yaml  # Entorno MiniGrid con objetivo discreto
│   │   ├── crafter.yaml
│   │   ├── dmc_vision.yaml
│   │   └── ...
│   └── model/                # Configs por tamaño de modelo
│       ├── _base_.yaml       # Arquitectura base
│       ├── size12M.yaml      # 12M parámetros
│       ├── size25M.yaml
│       └── ...
│
└── docs/
    ├── tensor_shapes.md      # Anotaciones de formas de tensores
    ├── envs.md               # Documentación de entornos
    └── docker.md             # Instrucciones Docker
```

---

## Flujo de datos en un paso de actualización

```
Observación (imagen + estado) + Misión (texto, opcional)
        │
        ▼
   MultiEncoder
        │ embed (B, T, E)
        ▼
      RSSM
   ┌─────────────────────────────────┐
   │  Deter: Block-GRU               │
   │  Posterior net: [deter, embed] → post_logit → z_post  │
   │  Prior net: deter → prior_logit → z_prior              │
   └─────────────────────────────────┘
        │ feat = [flatten(z_post), deter]
        ▼
   ┌─── KL Loss (dyn + rep) ───────┐
   │   TextEncoder KL (si texto)   │
   │   Representation Loss         │
   └───────────────────────────────┘
        │
        ▼
   Imagination Rollout (15 pasos en latent space)
        │
        ▼
   Reward Function: r(z, goal)    ◄── objetivo discreto
        │
        ▼
   Actor Loss (policy gradient)
   Critic Loss (temporal difference)
```

---

## Parámetros de configuración clave

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `rssm.stoch` | 32 | Número de grupos en el estado estocástico |
| `rssm.discrete` | 16 | Categorías por grupo (dimensión de z) |
| `rssm.deter` | 2048 | Dimensión del estado determinista |
| `model.rep_loss` | `r2dreamer` | Variante de loss de representación |
| `env.stochastic_classes` | 4 | Tamaño del objetivo discreto |
| `env.mission_text` | `False` | Activa el text encoder de misión |
| `trainer.imag_horizon` | 15 | Pasos de imagination rollout |
| `trainer.kl_free` | 1.0 | Free bits para KL (evita colapso) |
| `model.lr` | 4e-5 | Learning rate (LaProp) |

---

## Ejecución

```bash
# Entrenamiento básico con objetivo discreto
python3 train.py logdir=./logdir/test

# Con text encoder habilitado
python3 train.py \
  logdir=./logdir/goal_text/01 \
  env=random_goal \
  env.mission_text=True \
  model.rep_loss=r2dreamer \
  trainer.steps=500000

# Monitoreo con TensorBoard
tensorboard --logdir ./logdir
```

---

## Dependencias principales

- **PyTorch**: Framework de deep learning
- **TorchRL**: Replay buffer y utilidades de RL
- **Hydra**: Gestión de configuraciones
- **MiniGrid**: Entorno de cuadrícula para goal-conditioned RL
- **Crafter / DMC / Atari**: Entornos adicionales de benchmarking
