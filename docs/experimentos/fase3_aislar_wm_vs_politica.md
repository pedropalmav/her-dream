# Fase 3 — Aislar el problema: ¿es el WM o la política? (may 22 → may 30)

[← Índice](README.md)

## Idea

Separar el entrenamiento en dos etapas para atribuir la falla:

1. Entrenar **sólo el WM** con misiones aleatorias (`wm_only=True`,
   acciones random) → `wm_only_random_mission/01`.
2. **Congelarlo** (`freeze_wm=True`) y entrenar nada más actor/critic encima
   (`post_train.py`, commits `8946fa9`/`df9d1e3`).

Si con un WM aparentemente sano la política igual no aprende, **el WM no es el
cuello de botella** y el problema está en el reward / la fuente del goal.

## Corridas

Todas parten del WM congelado `wm_only_random_mission/01`, `env=fixed_goal`,
`goal_type=full`, 500k steps.

| Corrida | goal_sample | buffer | Score final | Máx | |
|---|---|---|---|---|:--:|
| `post_train_from_wm_only/her_buffer_goals` | buffer | HER | **-426** | -240 | 🟢 |
| `post_train_from_wm_only/normal_buffer_goals` | buffer | normal | **-497** | -245 | 🟢 |
| `post_train_from_wm_only/normalbuf_randomgoal/01` | random | normal | -1001 | -1001 | 🔴 |
| `post_train_from_wm_only/herbuf_textgoal/01` | text | HER | -1001 | -1001 | 🔴 |
| `distill_text_from_wm_only/01` | — (destila texto) | — | text_kl≈41.7 | — | ⚙ |
| `post_train_from_distill/01` | text | HER | -1001 | -1001 | 🔴 |

## La pista decisiva: `goal_sample=buffer` SÍ aprende

El hallazgo clave de esta fase es el contraste **dentro** de la misma receta de
WM congelado:

- Con **`goal_sample=buffer`** (goal = un `z` que el agente realmente visitó),
  el post-train **aprende** (-426 / -497). El WM congelado es perfectamente
  utilizable.
- Con **`goal_sample=text`** o **`random`**, se queda en -1001.

Esto **descarta** dos hipótesis a la vez:

1. **El WM no está roto por la política.** Congelado y todo, la política aprende
   cuando el goal es alcanzable (buffer).
2. **El problema tampoco es "post-training" como técnica.** Funciona; lo que no
   funciona es la *fuente del goal* cuando produce goals que el reward `full`
   nunca puede satisfacer (texto, random).

## El pipeline de destilación de texto

Para aislar aún más, se construyó (`distill_text.py`, commit `1985d02`) un modo
que destila el `TextEncoderGRU` contra el posterior de un WM congelado, con
actor/critic bypasseado. Responde "¿el text encoder *puede* imitar al
posterior?" sin contaminar con RL.

- `distill_text_from_wm_only/01` (200k): `text_kl` final ≈ 41.7.
- `post_train_from_distill/01` (500k): parte del checkpoint destilado,
  `goal_sample=text`, `goal_type=full` → **-1001**.

¿Por qué no aprende si la destilación "funciona"? El diagnóstico del 2026-06-02
(experimento `text_wm_alignment`) lo resuelve y es el corazón del proyecto →
[hallazgo_goal_type_full](hallazgo_goal_type_full.md): la destilación está muy
por encima del azar, pero `goal_type=full` exige match exacto de los 32 grupos,
con P≈4e-13. **El bloqueante es el reward, no la destilación.**

## Síntoma de "no recibe señal"

Las losses de las corridas pegadas en -1001 son consistentes con un actor que
**no diferencia entre acciones**:

| Corrida | `loss/policy` | `loss/value` |
|---|---|---|
| post-train random-goal | ≈ -0.0024 | 0.665 |
| post-train HER+text | ≈ -3.8e-4 | 0.899 |
| post-train desde destilado | ≈ -5e-4 | — |

Policy loss casi nula = no hay gradiente de política = nunca llega recompensa
positiva.

## Comandos

```bash
# 1) WM solo
python3 train.py logdir=./logdir/wm_only_random_mission/01 wm_only=True \
    env=fixed_goal env.mission_text=True trainer.steps=500000

# 2) post-train con WM congelado (goal del buffer → aprende)
python3 post_train.py logdir=./logdir/post_train_from_wm_only/her_buffer_goals \
    load_from=./logdir/wm_only_random_mission/01 freeze_wm=True \
    env.goal_sample=buffer buffer=her goal_type=full trainer.steps=500000
```
