# Posterior sin `h` (`obs_use_deter=False`) — no ayuda a `full`, pero **mejora** `row_by_row` en random_goal (jul 16 → jul 26)

[← Índice](README.md)

Ablación sobre la rama `feat/posterior-z-no-deter`: entrenar el WM con el
posterior condicionado **sólo en el embedding de la observación**, sin la parte
determinista `h` (`deter`). El `z` pasa a ser **puramente observacional**; el
prior `p(z|h)` y la transición determinista **no cambian**. La motivación era la
sospecha de la fuga de posición: si la posición del verde se filtra a `z`
*a través de `h`* ([random_goal_vs_fixed_goal](random_goal_vs_fixed_goal.md)),
quizá un `z` que no ve `h` produzca un latente más "limpio" y comparable como
goal.

**Resultado en dos partes:**

1. **Con `goal_type=full`, quitar `h` no mueve nada** — los tres post-trains
   quedan pegados al piso −1001, igual que sus contrapartes con `h`. Era
   esperable: el muro de `full` es la varianza de Gumbel (sample-vs-sample), no
   `h`.
2. **Con `goal_type=row_by_row` (la receta que sí da señal), quitar `h`
   *mejora*** — aprende ~2-3× más rápido y llega a un plateau ~110 puntos mejor
   que con `h`, sobre la misma seed y el mismo WM. **Este es el hallazgo del
   documento** (ver [§ El A/B con `row_by_row`](#el-ab-con-row_by_row-quitar-h-mejora)).

La lectura combinada: `h` no es lo que rompe el match exacto, pero para
**comparar goals** un `z` puramente observacional es más consistente — quitar `h`
ayuda justo cuando la recompensa deja ver la diferencia.

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

## El A/B con `row_by_row`: quitar `h` mejora

Los tres post-trains de arriba usaron `goal_type=full`, que garantiza −1001 con
o sin `h` y **enmascara** cualquier efecto de quitar `h`. La prueba que sí aísla
el efecto es correr la **única receta que aprende en `random_goal`** —`row_by_row`
+ imagination + HER ([artefacto_reward_exacto_gpu](artefacto_reward_exacto_gpu.md))—
sin `h`, y compararla 1:1 contra su espejo con `h` (`/03`): misma seed (3), mismo
WM randomstart (`agent_start_random`), mismo goal imaginado, HER, 1M pasos. La
única diferencia es `obs_use_deter`.

| Métrica (seed 3, 1M) | **SIN `h`** (`posttrain_randomstart_no_deter_rowbyrow/01`) | CON `h` (`...randomstart_goalimag_rowbyrow/03`) |
|---|---|---|
| `episode/score` ma25 final | **−180** | −290 |
| `episode/score` mejor episodio | **−22** | −116 |
| `eval_score` ma25 final | **−217** | −286 |
| `eval_score` mejor | **−121** | −220 |
| `train/rew` final | **−0.1** (~29/32 filas) | −0.2 (~25/32) |
| `episode/score` @143k (velocidad) | **−82** | −371 |

**Quitar `h` no solo no estorba — mejora en todo:**

- **Aprende ~2-3× más rápido**: llega a ≈ −82 a los 143k, donde el de `h` todavía
  va en −371.
- **Mejor plateau**: ma25 del score ≈ −180 (vs −290) y mejor episodio −22 (casi
  el techo ≈ 0).
- **Calza más filas en imaginación**: `train/rew` satura en ≈ −0.1 (≈29/32) en vez
  de −0.2 (≈25/32). Las ~7 filas "difusas" que topaban al de `h`
  ([artefacto §replica](artefacto_reward_exacto_gpu.md)) bajan a ~3 sin `h`.

### El WM sin `h` es un pelo *peor*, y aun así la política aprende mejor

El punto clave para el mecanismo: el WM-only sin `h` randomstart (item nuevo,
`wm_only_randomstart_no_deter`) queda **algo peor** que con `h`:

| WM randomstart | KL `dyn` (last10) | Barlow (last10) |
|---|---|---|
| con `h` (`wm_only_randomstart`) | 1.15 | 38.2 |
| **sin `h`** (`wm_only_randomstart_no_deter`) | **1.41** | **47.0** |

Es decir: **no es que el WM sea más nítido**. Con un posterior *menos* nítido, la
política goal-condicionada aprende *mejor*. Lo que cambia no es la calidad del WM
sino la **comparabilidad del `z` como goal**: al no apoyarse en `h`, el `z` del
estado y el `z` del goal imaginado quedan gobernados por el **mismo encoder
observacional**, así que sus argmax por fila coinciden más — justo lo que
`row_by_row` premia.

## Lecturas

- **Para `full`, quitar `h` no es palanca** (el muro es Gumbel/sample-vs-sample);
  para `row_by_row`, **sí lo es, y a favor**. La conclusión depende por completo
  de la forma de la recompensa — de ahí que probar solo con `full` (los tres
  primeros runs) llevara a la conclusión equivocada.
- **`h` ayuda a *jugar*, estorba para *comparar goals*.** Encaja con la ablación
  de crafter (quitar `h` cuesta ~13% de reward *controlando*,
  [diagnosticos §6](diagnosticos_espacio_representacion.md)) y con el hilo "info
  útil en `h` vs `z`" ([analisis_trayectorias_crafter](analisis_trayectorias_crafter.md)):
  la posición y buena parte del estado viven en `h`, y meter eso en el `z` que se
  compara lo vuelve más ruidoso como goal.
- **Pendiente para consolidar**: replicar el A/B `row_by_row` sin `h` en más
  seeds (hoy es 1 seed vs las 3 del lado con `h`), y ver el `eval_video` para
  confirmar el comportamiento (perseguir el `z` objetivo — que es el objetivo por
  diseño, no navegar al verde).

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
