# Fase 1 — La primera fila sí se puede alcanzar (abr 17 → abr 28)

[← Índice](README.md)

## Pregunta

¿El agente *puede siquiera* optimizar la recompensa goal-conditioned bajo este
reward function? Es decir, antes de preguntar si el goal puede venir del texto,
¿el aparato (misión textual + `TextEncoderGRU` entrenándose por KL +
actor/critic goal-conditioned) aprende algo cuando el goal es un `z` real?

## Setup (verificado contra `.hydra/config.yaml`)

> **QUÉ SE EVALÚA Y CÓMO.** Estas corridas miden si el aparato goal-conditioned
> **puede siquiera optimizar** el reward cuando el goal es un `z` **alcanzable**.
> Setup real (según la config guardada, `.hydra/config.yaml`): entorno
> **`random_goal`** y el reward compara **solo la primera fila** de `z`
> (`stoch[:, 0]`), **no** los 32 grupos (`full`) ni `fixed_goal`.

- `env.task: random-goal_` → **entorno `random_goal`** (el verde se mueve cada
  episodio), `env.mission_text: true`.
- **El goal es la primera fila de `z`** (`stoch[:, 0]`, shape `(K,)`), muestreada
  del **replay buffer** (un `z` que el agente realmente visitó). Esto es el
  reward de slot-0 *previo* a que existiera el config `goal_type`.
- **No hay `goal_type` ni `goal_sample` en la config**: ambos keys se
  introdujeron después (`goal_type` el 2026-04-24 en `ff35d2c` "add full z as
  goal"; `goal_sample=text` el 2026-05-07). La config de estas corridas
  (2026-04-28) es anterior, por lo que corrieron con el reward de **primera
  fila** hardcodeado (`her_buffer.py` devolvía `stoch[:, 0]`).
- **El `TextEncoderGRU` está presente y se entrena** (`text_kl` en las losses),
  así que la *dirección* del proyecto ya apuntaba al texto — por eso es fácil
  recordar estas corridas como "aprender con el text encoder". Pero el goal de
  **estas** corridas **no** sale del encoder: el muestreo de goal "observed z"
  (del buffer) se agregó la **misma tarde** (`09bb148`, 28-abr 17:47), justo
  antes de lanzarlas (20:27), mientras que "mission text as the goal"
  (`10666c6`) recién llegó el **1-may**, 3 días después. El encoder estaba
  entrenándose en paralelo; el goal, sin embargo, era un `z` del **buffer**.
- Dos seeds (3 y 4) para descartar suerte; 500k steps, `update_log_every=1000`.

## Resultado: **sí aprende**

| Corrida | Steps | Score final (últimos 10) | Máx score |
|---|---|---|---|
| `goal_dreamer_with_text/03` (seed 3) | 499k | **-357** | **-7** |
| `goal_dreamer_with_text/04` (seed 4) | 499k | **-416.8** | **-9** |
| `goal_dreamer_with_text/01` (seed 0) | 9k | -952 | -691 |

`/01` fue una corrida-puente demasiado corta (9k steps); las que importan son
`/03` y `/04`. Ambas seeds alcanzan **picos casi óptimos** (-7 y -9, sobre un
piso de -1001) y promedios finales muy por encima del piso. Es la **única
evidencia clara de aprendizaje goal-conditioned** del proyecto hasta que aparece
la recompensa `prob` (ver [recompensa_prob_funciona](recompensa_prob_funciona.md)).

Losses finales de la corrida 03 (referencia):

| Loss | Valor |
|---|---|
| `train/loss/dyn` | 1.33 |
| `train/loss/text_kl` | 11.51 |
| `train/loss/barlow` | 96.1 |
| `train/loss/policy` | ≈ 5.5e-4 |
| `train/loss/value` | 1.31 |

## Lecturas (a la luz de la corrección)

1. **El reward de slot-0 (primera fila) es entrenable, incluso en `random_goal`.**
   Que la primera fila baste vuelve la recompensa alcanzable: hay muchos estados
   cuyo `stoch[:, 0]` coincide con el goal, así que la política recibe señal.
2. **Esto encaja perfecto con el hallazgo central.** El mismo entorno
   `random_goal`, pero con reward **`full`** (32 grupos exactos), se queda en
   -1001 (ver [random_goal_vs_fixed_goal](random_goal_vs_fixed_goal.md)). La
   diferencia entre "aprende" y "no aprende" **nunca fue el entorno ni la fuente
   del goal, sino la *forma* del reward**: 1 fila vs 32 filas a la vez.
3. La `text_kl` se estaciona en ≈ 11.5 — no es chica. Queda la sospecha (que se
   confirma en Fase 2) de que ese residual es lo que rompe `goal_sample=text`
   cuando se intenta usar el encoder como fuente del goal.
4. Estas dos corridas son la **línea base** de "el aparato puede aprender".

## Comando (aproximado, reconstruido)

El comando original no quedó registrado con `goal_type`/`goal_sample` porque no
existían. Equivalente con el código actual sería un reward de primera fila
(`goal_type=first_row`) sobre `random_goal`:

```bash
python3 train.py logdir=./logdir/goal_dreamer_with_text/03 seed=3 \
    env=random_goal env.mission_text=True model.rep_loss=r2dreamer \
    goal_type=first_row \
    trainer.steps=500000 trainer.update_log_every=1000
```

> **Siguiente:** [Fase 2 — mover el goal al text encoder](fase2_goal_desde_texto.md).
