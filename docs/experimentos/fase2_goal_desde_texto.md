# Fase 2 — Hacer que el agente use de verdad el text encoder (may 7 → may 21)

[← Índice](README.md)

## Pregunta

Fase 1 muestra que aprende con goal del buffer (un `z` visitado, primera fila).
Pero el objetivo del proyecto es que
el goal dependa del **texto de la misión**. Se introduce `env.goal_sample=text`:
en cada episodio el goal se **muestrea del `TextEncoderGRU`** dada la misión, en
vez de venir del posterior del WM. ¿Sigue aprendiendo?

## Setup

- `env=fixed_goal`, `env.mission_text=True`, **`env.goal_sample=text`**.
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

## Lecturas

1. **Reemplazar la fuente del goal (WM → text encoder) rompe el aprendizaje.**
   Esto pone toda la atención en la *consistencia* entre el text encoder y el
   posterior del RSSM.
2. **El tamaño del RSSM no fue lo que rompió la cosa.** Reducir el espacio
   discreto (8×8) ayuda algo (picos) pero no resuelve.
3. **HER por sí solo tampoco salva el setup.** En 32×16, HER y normal dan el
   mismo -1001.
4. **Lección de ingeniería** (`e818b7e`, 21 may): la misión se pasa por dentro
   como **int** y se one-hottea recién en el forward del encoder. Pasar el
   one-hot completo del vocabulario por todo el grafo era prohibitivo en memoria.
   El one-hot vive *adentro* del encoder, no en la observación.

## La hipótesis que abre Fase 3

La sospecha natural es que el text encoder produce un `z` *demasiado ruidoso o
inconsistente* respecto del posterior del WM, y el reward function nunca se
dispara. Pero, ¿es el text encoder, el WM, o la política? Para separarlo nace el
post-training con WM congelado → [Fase 3](fase3_aislar_wm_vs_politica.md).

> **Spoiler.** El verdadero culpable no es ninguno de esos tres por separado,
> sino la **forma de la recompensa** `goal_type=full`: ver
> [hallazgo_goal_type_full](hallazgo_goal_type_full.md). Pero eso recién se
> diagnostica el 2026-06-02.

## Comando

```bash
python3 train.py logdir=./logdir/text_goal_sample_her_buffer/01 \
    env=fixed_goal env.mission_text=True env.goal_sample=text \
    buffer=her goal_type=full trainer.steps=500000
```
