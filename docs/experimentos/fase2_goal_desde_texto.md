# Fase 2 — Hacer que el agente use de verdad el text encoder (may 7 → may 21)

[← Índice](README.md)

## Pregunta

Fase 1 muestra que aprende con goal del buffer (un `z` visitado, primera fila).
Pero el objetivo del proyecto es que
el goal dependa del **texto de la misión**. Se introduce `env.goal_sample=text`:
en cada episodio el goal se **muestrea del `TextEncoderGRU`** dada la misión, en
vez de venir del posterior del WM. ¿Sigue aprendiendo?

## Setup (verificado contra `.hydra/config.yaml`)

> ⚠️ **Corrección importante.** Las cuatro corridas se lanzaron en
> **`random_goal`** (`task: random-goal_`), **no** en `fixed_goal`. Esto es un
> confound central para interpretar el resultado (ver abajo).

- **`env=random_goal`**, `env.mission_text=True`, **`env.goal_sample=text`**.
- `goal_type=full`, 500k steps.
- Se barre la grilla **buffer (HER vs normal) × tamaño de RSSM (32×16 vs 8×8)**
  para descartar que sea un problema de capacidad o de re-etiquetado.

## Resultado: **todo se cae a -1001**

| Corrida | RSSM | buffer | Score final | Máx |
|---|---|---|---|---|
| `text_goal_sample_her_buffer/01` | 32×16 | HER | -1001 | -1001 |
| `text_goal_sample_normal_buffer/01` | 32×16 | normal | -1001 | -1001 |
| `text_goal_sample_8x8_goal_her_buffer/01` | 8×8 | HER | -998 | -121 |
| `text_goal_sample_8x8_normal_buffer/01` | 8×8 | normal | -1000 | -811 |

Las cuatro combinaciones quedan pegadas al piso. El 8×8 con HER logra **picos
aislados** (-121) pero la mayoría de los episodios siguen en -1001.

## ¿Encoder malo o entorno random_goal? — el confound

La lectura "natural" (en su momento) fue: *el text encoder produce un `z`
demasiado ruidoso y por eso el reward nunca se dispara*. Pero al verificar que
estas corridas eran **random_goal** + **`goal_type=full`**, el fallo queda
**sobre-determinado** y ya **no** se puede atribuir limpiamente al encoder:

- **El entorno random_goal con `full` falla por sí solo**, aun con un goal del
  **buffer** (no de texto): `random_goal/...frozenwm_normalbuf_goalbuf` y
  `...herbuf_goalbuf` se quedan en -1001
  ([random_goal_vs_fixed_goal](random_goal_vs_fixed_goal.md)). La posición del
  verde se filtra a `z`, así que un goal de otra posición es inalcanzable con
  `full`. Esto pasa **independientemente de la calidad del text encoder**.
- De hecho, random_goal + `full` falla incluso con goal de **imaginación**
  (alcanzable por construcción) y reward denso `prob`
  ([recompensa_prob](recompensa_prob_funciona.md)). El entorno es el factor duro.
- Como contraste, random_goal **sí aprende** con reward de **primera fila**
  (Fase 1, `goal_dreamer_with_text` → -357/-417). O sea, random_goal no está
  condenado; lo que lo rompe es `full`.

**Conclusión:** estas cuatro corridas mezclan dos causas conocidas de -1001
(entorno random_goal + reward `full`) con la hipótesis del encoder, así que **no
aíslan** si el text encoder es el problema. El encoder *sí* es difuso (residual
de KL ≈ 11.5, colapsa 94 misiones en 7/16 clases del slot 0; ver
[hallazgo_goal_type_full](hallazgo_goal_type_full.md)), pero eso es un factor
adicional, no la causa demostrada aquí. **Para aislar el encoder haría falta
re-correr en `fixed_goal` con un reward alcanzable (`first_row`/`prob`).**

## Otras lecturas

1. **El tamaño del RSSM no fue lo que rompió la cosa.** Reducir el espacio
   discreto (8×8) ayuda algo (picos -121) pero no resuelve.
2. **HER por sí solo tampoco salva el setup.** En 32×16, HER y normal dan el
   mismo -1001.
3. **Lección de ingeniería** (`e818b7e`, 21 may): la misión se pasa por dentro
   como **int** y se one-hottea recién en el forward del encoder. Pasar el
   one-hot completo del vocabulario por todo el grafo era prohibitivo en memoria.
   El one-hot vive *adentro* del encoder, no en la observación.

## La hipótesis que abre Fase 3

La sospecha natural es que el text encoder produce un `z` *demasiado ruidoso o
inconsistente* respecto del posterior del WM, y el reward function nunca se
dispara. Pero, ¿es el text encoder, el WM, o la política? Para separarlo nace el
post-training con WM congelado → [Fase 3](fase3_posttrain_wm_aleatorio.md).

> **Spoiler.** Un factor de mucho peso es la **forma de la recompensa**
> `goal_type=full` (match exacto sobre los 32 grupos), que recién se diagnostica
> el 2026-06-02: ver [hallazgo_goal_type_full](hallazgo_goal_type_full.md). Pesa
> **sobre todo cuando el goal viene del text encoder**, y también en
> `random_goal` con goal del buffer. Pero **no es universal**: con goal del
> **buffer**, `fixed_goal` aprende bien con `full` (Fase 3). O sea, `full` es
> muy importante, pero **no es la única causa** ni explica por sí solo todo el
> gap fixed↔random.

## Comando

```bash
python3 train.py logdir=./logdir/text_goal_sample_her_buffer/01 \
    env=random_goal env.mission_text=True env.goal_sample=text \
    buffer=her goal_type=full trainer.steps=500000
```
