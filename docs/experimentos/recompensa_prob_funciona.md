# El desenlace — recompensa densa `prob` + goal de imaginación SÍ aprende

[← Índice](README.md)

Tras confirmar que `goal_type=full` (match exacto sample==sample sobre 32 grupos)
es inalcanzable por construcción — tanto por la divergencia text↔WM
([hallazgo central](hallazgo_goal_type_full.md)) como por el ruido de Gumbel del
muestreo ([random_goal](random_goal_vs_fixed_goal.md)) — la salida es cambiar la
**forma de la recompensa**: de un indicador binario inalcanzable a una **señal
densa**.

## La recompensa `prob`

`goal_type=prob` (`prob_reward` en `rewards.py:131`):

```python
def prob_reward(dist, goal):
    # dist = distribución prior sobre z en el estado imaginado
    # goal = one-hot objetivo (S, K)
    return dist.log_prob(goal).exp().unsqueeze(-1)   # ∈ [0, 1]
```

Es la **probabilidad de muestrear exactamente el goal** bajo la distribución del
estado actual — el producto sobre los 32 grupos de la probabilidad de cada
categoría objetivo. **Densa, sin umbral**, en [0, 1]:

- Reemplaza el "¿coinciden las 32 muestras?" (que da -1 casi siempre) por
  "¿qué tan probable es este goal bajo mi estado?".
- Cuanto más cerca está el agente del estado-goal, más alta la probabilidad →
  gradiente útil en todo momento, sin necesidad de un acierto exacto e
  improbabilísimo.
- Elimina la varianza de Gumbel del lado del estado: no muestrea, evalúa una
  densidad.

Se combina con **`goal_sample=imagination`**: el goal se genera rodando el WM
`imag_horizon` pasos desde la obs inicial con acciones aleatorias, así es
**alcanzable por construcción** desde el episodio actual.

## Resultados: aprende en ambos entornos

| Corrida | Env | goal_sample | goal_type | Score final | Máx | |
|---|---|---|---|---|---|:--:|
| `post_train_from_wm_only/04_imag_prob/01` | **fixed_goal** | imagination | prob | **+331** | +727 | 🟢 |
| `random_goal/...frozenwm_normalbuf_goalimag_prob/01` | **random_goal** | imagination | prob | **0** | +73 | 🟢 |

Comparación directa contra la misma fuente de goal con `full`:

| Env | goal_sample | `full` | `prob` |
|---|---|---|---|
| random_goal | imagination | **-1001** 🔴 | **0** 🟢 |

El cambio de `full` → `prob`, **manteniendo todo lo demás igual** (mismo WM
congelado, mismo goal imaginado, mismo buffer normal), es lo que desbloquea el
aprendizaje. Aísla limpiamente que el problema era la **forma de la recompensa**,
no la fuente del goal ni el WM.

> Estos scores positivos (sobre todo el +331/+727 de fixed_goal y el salto de
> -1001 a 0 en random_goal) son el resultado más fuerte del proyecto y **aún no
> estaban en `bitacora_nano/general.md`** al consolidar esta documentación
> (la entrada del 06-16 cierra con el spoiler de que imagination+`full` falla,
> antes de correr la variante `prob`).

![Score imagination + prob](assets/score_imag_prob.png)

## Lectura

1. **El reward latente `z` sí es entrenable** si se le da forma densa. El muro
   nunca fue "comparar `z`", fue exigir **igualdad exacta de muestras** sobre un
   espacio de alta dimensión y estocástico.
2. **`prob` + `imagination` se complementan**: la imaginación garantiza que
   exista *algún* camino al goal; `prob` garantiza que el agente reciba señal
   *gradiente* a lo largo de ese camino en vez de un premio binario casi nunca
   disparado.
3. Queda pendiente cerrar el círculo original del proyecto: ¿funciona también
   con **`goal_sample=text`** (goal derivado de la misión) usando `prob` en vez
   de `full`? Es la continuación natural — la receta que funciona ya está
   identificada, falta enchufarle la fuente de goal textual.

## Comandos

```bash
# fixed_goal — imagination + prob (item 22 de execution_commands.md)
bash scripts/post_train.sh \
    load_from=./logdir/wm_only_random_mission/01 \
    logdir=./logdir/post_train_from_wm_only/04_imag_prob/01 \
    freeze_wm=True wm_only=False mission_text=True \
    env=fixed_goal env.goal_sample=imagination \
    buffer=normal goal_type=prob seed=1 trainer.steps=500000

# random_goal — imagination + prob (item 23)
bash scripts/post_train.sh \
    load_from=./logdir/random_goal/wm_only_randomgoal/01 \
    logdir=./logdir/random_goal/posttrain_frozenwm_normalbuf_goalimag_prob/01 \
    freeze_wm=True wm_only=False env=random_goal seed=1 mission_text=False \
    env.goal_sample=imagination buffer=normal goal_type=prob trainer.steps=500000
```
