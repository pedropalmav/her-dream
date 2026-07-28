# Posterior sin `h` (`obs_use_deter=False`) — quitar la historia del `z` no destraba `full` en random_goal (jul 16 → jul 25)

[← Índice](README.md)

Ablación sobre la rama `feat/posterior-z-no-deter`: entrenar el WM con el
posterior condicionado **sólo en el embedding de la observación**, sin la parte
determinista `h` (`deter`). El `z` pasa a ser **puramente observacional**; el
prior `p(z|h)` y la transición determinista **no cambian**. La motivación era la
sospecha de la fuga de posición: si la posición del verde se filtra a `z`
*a través de `h`* ([random_goal_vs_fixed_goal](random_goal_vs_fixed_goal.md)),
quizá un `z` que no ve `h` produzca un latente más "limpio" y comparable, y el
match `full` deje de romperse en `random_goal`.

**Resultado corto: no. Con `goal_type=full`, quitar `h` del posterior no mueve
nada** — los tres post-trains quedan pegados al piso −1001, igual que sus
contrapartes con `h`. Ni el WM se rompe ni mejora la comparabilidad del goal.

## El cambio (`model.rssm.obs_use_deter=False`)

`_obs_net` calcula los logits del posterior. Por defecto recibe `[deter, embed]`;
con `obs_use_deter=False` recibe **sólo `embed`**. Es la misma ablación
`post_use_deter` que en Crafter costaba ~13% de reward
([diagnosticos §6](diagnosticos_espacio_representacion.md)), ahora aplicada al WM
de `random_goal` como intento de destrabar el gap.

> **Nota operativa (del `execution_commands.md`, items 29-32).** `load_from` sólo
> carga el `state_dict`; el modelo se **reconstruye con la config actual**. Por
> eso hay que **repetir `model.rssm.obs_use_deter=False` también en cada
> post-train** — si se omite, `load_state_dict` falla por mismatch de dimensiones
> en `_obs_net` (con `h`, la entrada es `deter+embed`; sin `h`, sólo `embed`).
> Verificado en los cuatro `.hydra/overrides.yaml`.

## Corridas (todas `random_goal`, seed 1, `goal_type=full` por default, `compile=True`, 500k)

| # | Corrida | Fase | goal_sample | buffer | Score last10 | Máx ep. | `train/rew` (last10 / máx) | |
|---|---|---|---|---|---|---|---|:--:|
| 29 | `wm_only_randomgoal_no_deter/01` | WM only | random | normal | −1001 (esperado) | — | — | ⚙ solo WM |
| 30 | `posttrain_frozenwm_no_deter_herbuf_goalbuf/01` | post-train | buffer | HER | **−1001** | −997 | −0.97 / −0.91 | 🔴 |
| 31 | `posttrain_frozenwm_no_deter_normalbuf_goalbuf/01` | post-train | buffer | normal | **−1001** | −969 | −1.00 / −0.99 | 🔴 |
| 32 | `posttrain_frozenwm_no_deter_normalbuf_goalimag/01` | post-train | imagination | normal | **−1001** | −556 | −0.98 / −0.96 | 🔴 |

En los tres, `train/rew` (la señal de gradiente del actor/critic) se queda en
≈ −1 y `loss/policy` colapsa a ≈ 0: **nunca llega recompensa positiva**, no hay
gradiente de política. El `episode/score` roza picos aislados (el goalimag llegó
a −556 en un episodio suelto), pero la media móvil nunca despega del piso.

## El WM sin `h` es sano (no se rompió)

El WM-only sin `h` (item 29) entrena bien; sólo queda **un pelo peor** que el
WM con `h` (item 17, `wm_only_randomgoal/01`), como es de esperar al quitarle
capacidad al posterior:

| WM-only | KL `dyn`/`rep` (last10) | Barlow (last10) |
|---|---|---|
| con `h` (`wm_only_randomgoal`) | 1.24 | 42.8 |
| **sin `h`** (`wm_only_randomgoal_no_deter`) | **1.50** | **48.2** |

El KL piso sube ~0.26 nats y el Barlow ~13% — el posterior sin `h` es algo menos
nítido, pero claramente **no está roto**. `eval_score` es −1001 en ambos, lo
esperado para un WM-only con acciones aleatorias. Así que el −1001 de los
post-trains **no** se explica por un WM degenerado.

## Comparación directa con las contrapartes con `h` (items 18/19/20)

Mismo entorno, misma seed, mismo goal/buffer — la única diferencia es
`obs_use_deter`:

| goal_sample / buffer | con `h` (18/19/20), máx ep. | sin `h` (30/31/32), máx ep. |
|---|---|---|
| buffer / HER | −637 | −997 |
| buffer / normal | −942 | −969 |
| imagination / normal | −1000 | −556 |

**Ambas familias fallan idéntico**: todas −1001 de media, todas con `train/rew` ≈
−1. Las diferencias de máximo son ruido de episodios sueltos (ninguna sostiene
una media móvil por encima del piso). Quitar `h` del posterior **no cambia el
veredicto** para `full` en `random_goal`.

## Por qué no ayudó (y por qué era esperable)

1. **`full` es sample-vs-sample: el muro es Gumbel, no `h`.** Aunque `z` fuera
   perfectamente "limpio" de posición, `full` compara dos muestras one-hot
   independientes del mismo Gumbel-Softmax, y eso ya matchea 0% *incluso para el
   mismo estado* (columna C de `goal_reachability`,
   [random_goal_vs_fixed_goal](random_goal_vs_fixed_goal.md)). Sacar `h` no toca
   esa varianza de muestreo.
2. **Quitar `h` no quita la posición del `z` — puede meter más.** La posición del
   verde está **en la observación** (el cuadrado es visible). Un posterior que ya
   no se apoya en `h` debe cargar en `z` *todo* lo que antes tomaba de `deter`,
   así que el `z` observacional podría codificar **más** posición, no menos —
   empeorando, si acaso, el caso cross-posición de `goalbuf`. La hipótesis "limpia
   el `z`" era, en retrospectiva, al revés.
3. **La contraparte con `h` ya fallaba por la forma de la recompensa**, no por
   `h`. Esta ablación mueve la variable equivocada: el cuello de botella
   documentado es `full`/sample-vs-sample
   ([hallazgo_goal_type_full](hallazgo_goal_type_full.md)), y estos runs lo dejan
   fijo.

## El hueco: `row_by_row` sin `h` **no se corrió**

Los tres post-trains usaron `goal_type=full` (el default, no se sobrescribió).
La **única receta que aprende en `random_goal`** —`row_by_row` + imagination +
HER— nunca se probó sobre el WM sin `h`
([artefacto_reward_exacto_gpu](artefacto_reward_exacto_gpu.md)). Así que esta
ablación **no responde** su pregunta natural: *¿un `z` puramente observacional
ayuda o estorba cuando la recompensa sí da señal?* Con `full` todo es −1001 por
la razón de siempre, y eso enmascara cualquier efecto real de quitar `h`.

## Lecturas

- **Quitar `h` del posterior no es una palanca para el gap fixed↔random** bajo
  `full`. El WM sigue sano; el −1001 es el mismo muro de la recompensa.
- La ablación, tal como se corrió, **está confundida** por `goal_type=full`
  (igual que la Fase 2 estaba confundida por random+full): no aísla el efecto de
  `obs_use_deter` porque la recompensa ya garantiza −1001 con o sin `h`.
- Para que la ablación signifique algo hay que correrla con una recompensa que
  **sí da gradiente** (`row_by_row`), y comparar contra el WM con `h` bajo esa
  misma receta.

## Comandos (del `execution_commands.md`, items 29-32)

```bash
# 29) WM only, posterior sin h
bash scripts/train.sh \
    logdir=./logdir/random_goal/wm_only_randomgoal_no_deter/01 \
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=random buffer=normal wm_only=True \
    model.rssm.obs_use_deter=False \
    env.steps=500000 trainer.update_log_every=1000

# 30-32) post-train congelado sobre ese WM (repetir obs_use_deter=False!)
#   30: env.goal_sample=buffer      buffer=her
#   31: env.goal_sample=buffer      buffer=normal
#   32: env.goal_sample=imagination buffer=normal
bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomgoal_no_deter/01 \
    logdir=./logdir/random_goal/posttrain_frozenwm_no_deter_herbuf_goalbuf/01 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=buffer buffer=her \
    model.rssm.obs_use_deter=False \
    env.steps=500000 trainer.update_log_every=1000
```
