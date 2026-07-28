# Experimentos — Goal-conditioned R2-Dreamer con goal latente `z`

Documentación versionada de los experimentos del proyecto: qué se corrió, qué
resultó, **qué sirve y qué no**, con gráficos. Es la contraparte en `git` de la
bitácora personal `bitacora_nano/` (que está en `.gitignore`): aquí se consolida
lo reproducible y presentable.

> **Pregunta de investigación.** ¿Se puede aprender una política
> goal-conditioned sobre un Dreamer cuyo *goal* es un latente discreto `z`
> (32 grupos × 16 categorías), comparándolo contra el `z` del estado imaginado?
> El goal puede venir del buffer, de un encoder de texto que describe la misión,
> o de la imaginación del world-model.

## Cómo leer esta carpeta

| Documento | Contenido |
|---|---|
| [`fase1_primera_row_alcanzable.md`](fase1_primera_row_alcanzable.md) | Prueba preliminar: la **primera fila** del `z` **sí se puede alcanzar** y la política aprende cuando el goal sale del posterior del WM (seeds 3/4 → -357/-417). |
| [`fase2_goal_desde_texto.md`](fase2_goal_desde_texto.md) | Mover la fuente del goal al text-encoder rompe el aprendizaje (todo a -1001). **Pendiente:** rehacer estos experimentos en **fixed_goal** (se corrieron en random_goal, que ya falla por sí solo y confunde el resultado). |
| [`fase3_posttrain_wm_aleatorio.md`](fase3_posttrain_wm_aleatorio.md) | Post-train sobre un WM entrenado con acciones aleatorias: **congelar el WM sí ayuda** a los resultados. **Falta** una corrida **sin** congelar el WM para medir el salto de diferencia. |
| [`hallazgo_goal_type_full.md`](hallazgo_goal_type_full.md) | **Un bloqueante de mucho peso** (sobre todo con goal de **texto**/random): `goal_type=full` (sample==sample sobre 32 grupos) es inalcanzable por construcción (P≈4e-13). No universal: con goal del **buffer**, fixed_goal sí aprende con `full`. |
| [`random_goal_vs_fixed_goal.md`](random_goal_vs_fixed_goal.md) | Por qué post-train funciona en fixed_goal y falla en random_goal: la posición del verde vive en `z`/`deter`. Incluye `goal_sample=imagination`. |
| [`recompensa_prob_funciona.md`](recompensa_prob_funciona.md) | Recompensa densa `prob` + goal de imaginación → aprende en **fixed_goal** (+331), pero **no** en random_goal (se clava en el piso 0). El gap fixed↔random sigue abierto. |
| [`artefacto_reward_exacto_gpu.md`](artefacto_reward_exacto_gpu.md) | **Bug de cómputo + primer aprendizaje en random_goal.** El `==` de floats de los rewards de match exacto se rompió en el paso GPU entre el 8 y el 26 de junio (train/rew clavado en -1 = gradiente cero); con el fix argmax, la receta rowbyrow+imagination del item 27 **aprende en random_goal**, replicado en 3/3 seeds con plateau ≈ -290 a 1M. Matiz: alcanza el goal latente, no el cuadrado verde. |
| [`posterior_sin_deter.md`](posterior_sin_deter.md) | **Ablación `obs_use_deter=False`** (posterior sin `h`): con `full` no mueve nada (sigue -1001, items 29-32), pero con **`row_by_row` quitar `h` MEJORA** — mismo seed/WM/goal, aprende ~2-3× más rápido y plateau ma25 ≈ **-180 vs -290** con `h`. Y el WM sin `h` es un pelo *peor* → no es nitidez, es que el `z` observacional es más comparable como goal. |
| [`analisis_trayectorias_crafter.md`](analisis_trayectorias_crafter.md) | Inspección de los `z` a lo largo de rollouts reales: en ambientes **complejos** (crafter) los `z` son **muy estocásticos** (entropía ≈1.96/4 bits, 48% de grupos difusos), mientras que en **fixed_goal** tienden a ser **mucho más deterministas**. Abre la pregunta de cuánta información útil vive en `z` vs en `h`. |
| [`diagnosticos_espacio_representacion.md`](diagnosticos_espacio_representacion.md) | Las herramientas de `experiments/` (estocasticidad del WM/text, consistencia del posterior). |
| [`papers_sugeridos.md`](papers_sugeridos.md) | Literatura relacionada (clásica + reciente 2024-2025) mapeada a cada problema/receta del proyecto. La lectura común: **abandonar el match exacto en latente** en favor de distancias aprendidas y goals con incertidumbre explícita. |

## La historia en una línea

El reward `z(estado) == z(goal)` **es entrenable** (Fase 1; y `goal_type=full`
con goal del buffer aprende en fixed_goal → el match exacto **no** es imposible).
Lo que lo rompe es exigir match **exacto sobre los 32 grupos** cuando el goal
viene de una fuente que produce goals inalcanzables (texto, random,
cross-posición). Una recompensa **densa** (`goal_type=prob`) sobre un goal
**alcanzable por construcción** (`goal_sample=imagination`) **no desbloquea**
nada nuevo en **fixed_goal** —que **ya aprendía desde antes** (post-train con
`full`+buffer llega a ≈-426/-497)— y **tampoco** mueve a **random_goal**, que
sigue **clavado en el piso 0**. Persiste un **gap grande entre fixed_goal y random_goal** que es la
pregunta abierta central: la misma receta funciona en uno y falla en el otro.

**Giro (jul 2026):** parte del gap era un **bug de cómputo**, no el entorno — el
`==` de floats de los rewards de match exacto se rompió en el paso de
entrenamiento GPU a mediados de junio (train/rew clavado en -1 → gradiente
cero). Con el fix argmax (`025f3b3`), **random_goal aprende por primera vez**
(`row_by_row` + imagination), y quedó **replicado en 3/3 seeds** (jul 7-9):
trayectorias casi calcadas (ma25 ≈ -395/-439/-395 @249k) y **plateau en
score ≈ -290** a 1M de pasos — el cuello de botella ya no son los pasos
(`train/rew` satura en ≈ -0.21 ≈ 25/32 filas). El matiz honesto: aprende a
alcanzar el **goal latente imaginado**, no a ir al cuadrado verde (el goal no
se lo pide); conectar el goal con la tarea real (`goal_sample=image`) es el
siguiente eslabón. La rama `prob`, en cambio, queda **descartada en
random_goal**: con HER + el WM randomstart (seeds 1-2) sigue plana en el piso 0,
así que ni el re-etiquetado ni un WM más entrenado eran lo que le faltaba. Ver
[artefacto_reward_exacto_gpu](artefacto_reward_exacto_gpu.md).

## Tabla maestra de corridas

Score = media de los últimos 10 episodios (piso = **-1001**, el time-limit sin
alcanzar el goal; en `crafter` la escala es otra). 🟢 = aprende, 🔴 = pegado al piso.

| Corrida | Env | goal_sample | goal_type | buffer | Steps | Score final | Máx | |
|---|---|---|---|---|---|---|---|:--:|
| `goal_dreamer_with_text/03` (seed 3) | random | buffer (1ª fila) | first_row† | normal | 499k | **-357** | -7 | 🟢 |
| `goal_dreamer_with_text/04` (seed 4) | random | buffer (1ª fila) | first_row† | normal | 499k | **-417** | -9 | 🟢 |
| `goal_dreamer_with_text/01` (seed 0) | random | buffer (1ª fila) | first_row† | normal | 9k | -952 | -691 | ⏸ corta |
| `text_goal_sample_her_buffer/01` | random§ | text | full | HER | 499k | -1001 | -1001 | 🔴 |
| `text_goal_sample_normal_buffer/01` | random§ | text | full | normal | 499k | -1001 | -1001 | 🔴 |
| `text_goal_sample_8x8_goal_her_buffer/01` (RSSM 8×8) | random§ | text | full | HER | 499k | -998 | -121 | 🔴 |
| `text_goal_sample_8x8_normal_buffer/01` (RSSM 8×8) | random§ | text | full | normal | 499k | -1000 | -811 | 🔴 |
| `wm_only_random_mission/01` | fixed | — | — | — | 499k | -1001 (esperado) | — | ⚙ solo WM |
| `post_train_from_wm_only/her_buffer_goals` | fixed | buffer | full | HER | 499k | **-426** | -240 | 🟢 |
| `post_train_from_wm_only/normal_buffer_goals` | fixed | buffer | full | normal | 499k | **-497** | -245 | 🟢 |
| `post_train_from_wm_only/normalbuf_randomgoal/01` | fixed | random | full | normal | 499k | -1001 | -1001 | 🔴 |
| `post_train_from_wm_only/herbuf_textgoal/01` | fixed | text | full | HER | 499k | -1001 | -1001 | 🔴 |
| `distill_text_from_wm_only/01` | fixed | — | full | — | 199k | (text_kl≈41.7) | — | ⚙ destila texto |
| `post_train_from_distill/01` | fixed | text | full | HER | 499k | -1001 | -1001 | 🔴 |
| `random_goal/wm_only_randomgoal/01` | random | — | — | — | 499k | -1001 (esperado) | — | ⚙ solo WM |
| `random_goal/...frozenwm_herbuf_goalbuf/01` | random | buffer | full | HER | 499k | -1001 | -637 | 🔴 |
| `random_goal/...frozenwm_normalbuf_goalbuf/01` | random | buffer | full | normal | 499k | -1001 | -942 | 🔴 |
| `random_goal/...frozenwm_normalbuf_goalimag/01` | random | imagination | full | normal | 499k | -1001 | -1000 | 🔴 |
| **`post_train_from_wm_only/04_imag_prob/01`** | **fixed** | **imagination** | **prob**‡ | normal | 499k | **+331** | +727 | 🟢 |
| **`random_goal/...frozenwm_normalbuf_goalimag_prob/01`** | **random** | **imagination** | **prob**‡ | normal | 499k | **0** (piso) | +73 | 🔴 |
| `random_goal/...randomstart_goalimag_full/01` (item 27) | random | imagination | full | HER | 499k | -1001 | -999 | 🔴 |
| `random_goal/...randomstart_goalimag_rowbyrow/01` | random | imagination | row_by_row¶ | HER | 499k | -686 (plano) | — | ⚠️ artefacto |
| **`random_goal/...rowbyrow_fixedrew/01`** (seed 1) | **random** | **imagination** | **row_by_row**¶ | HER | 249k× | **-396 ↗ subiendo** | -114 | 🟢 **primero en random** |
| **`random_goal/...randomstart_goalimag_rowbyrow/02`** (seed 2) | **random** | **imagination** | **row_by_row**¶ | HER | 499k | **-316 ↗ subiendo** | -102 | 🟢 réplica |
| **`random_goal/...randomstart_goalimag_rowbyrow/03`** (seed 3) | **random** | **imagination** | **row_by_row**¶ | HER | 999k | **-290 (plateau ~700k)** | -116 | 🟢 réplica, 1M |
| `random_goal/...randomstart_goalimag_prob/{01,02}` (seeds 1-2) | random | imagination | prob‡ | HER | 499k | 0 (piso) | 12 | 🔴 |
| `random_goal/wm_only_randomgoal_no_deter/01`ⁿ (item 29) | random | — | — | — | 500k | -1001 (esperado) | — | ⚙ solo WM |
| `random_goal/...no_deter_herbuf_goalbuf/01`ⁿ (item 30) | random | buffer | full | HER | 500k | -1001 | -997 | 🔴 |
| `random_goal/...no_deter_normalbuf_goalbuf/01`ⁿ (item 31) | random | buffer | full | normal | 500k | -1001 | -969 | 🔴 |
| `random_goal/...no_deter_normalbuf_goalimag/01`ⁿ (item 32) | random | imagination | full | normal | 500k | -1001 | -556 | 🔴 |
| **`random_goal/...randomstart_no_deter_rowbyrow/01`**ⁿ (seed 3) | **random** | **imagination** | **row_by_row**¶ | HER | 999k | **-180 (ma25) ↗** | -22 | 🟢 **mejor que con `h`** |
| `original_wm_crafter/02` (baseline vanilla) | crafter | — | — | — | 1.01M | eval≈10.9 | 14.1 | 🟢 WM juega |
| `z_without_history_wm_crafter/01` (ablación) | crafter | — | — | — | 1.01M | eval≈8.7 | 12.1 | 🟢 −13% |

> † Las corridas `goal_dreamer_with_text/*` (abr-28) son **anteriores** a que
> existieran los keys `goal_type`/`goal_sample`: corrieron con el reward de
> **primera fila** hardcodeado (`stoch[:, 0]`), no con `full`. El equivalente
> actual es `goal_type=first_row`. Verificado contra su `.hydra/config.yaml`.
> Detalle en [fase1](fase1_el_setup_funciona.md).
>
> ‡ Las corridas `goal_type=prob` están en **otra escala**: el score es la suma
> de la recompensa de probabilidad ∈ [0,1], con **piso 0** (no -1001). Ahí
> **0 = no aprende** (mínimo), no éxito. Por eso no son comparables con las demás
> y van en su propia figura ([recompensa_prob](recompensa_prob_funciona.md)).
>
> § Las corridas `text_goal_sample_*` (Fase 2) se lanzaron en **`random_goal`**
> (verificado vs `.hydra/config.yaml`), no en fixed_goal. Su -1001 está
> **confundido**: random_goal + `full` ya falla por sí solo (sin texto), así que
> no aíslan al text encoder. Detalle en [fase2](fase2_goal_desde_texto.md).
>
> ⁿ Las corridas `...no_deter...` (items 29-32) usan el **posterior sin `h`**
> (`model.rssm.obs_use_deter=False`, rama `feat/posterior-z-no-deter`): el `z` se
> condiciona sólo en el embed de la obs. Todas con `goal_type=full` → -1001, igual
> que sus contrapartes con `h` (items 18-20): quitar `h` **no** destraba `full`.
> Detalle en [posterior_sin_deter](posterior_sin_deter.md).
>
> ¶ Con `row_by_row` el score es denso (suma de `(filas/32)−1` por paso): el
> piso sigue siendo ≈-1001 pero una política aleatoria ronda -700, así que "no
> aprende" = plano ~-690, no -1001. El run del 26 jun está **contaminado por el
> artefacto del `==`** (train/rew=-1 → gradiente cero); el `fixedrew` es la
> misma receta con el fix argmax (`025f3b3`) y **× murió a 249k/500k** (corte
> silencioso). Los `/02` (seed 2, 500k) y `/03` (seed 3, 1M) son sus
> relanzamientos completos. Ver
> [artefacto_reward_exacto_gpu](artefacto_reward_exacto_gpu.md).

## Figuras de conjunto

Generadas por [`scripts/plot_runs.py`](scripts/plot_runs.py) a partir de los
eventos de TensorBoard. Re-ejecutar con `uv run python3
docs/experimentos/scripts/plot_runs.py` cuando lleguen corridas nuevas.

### Qué sirve y qué no — score final por corrida

Solo corridas con reward ∈ {-1, 0} (piso -1001), comparables entre sí. Las
corridas `prob` están en otra escala y se grafican aparte (abajo).

![Score final por corrida](assets/score_final_barras.png)

### Curvas de entrenamiento por fase

![Curvas por fase](assets/curvas_por_fase.png)

### Corridas `goal_type=prob` (escala propia, piso 0)

Aquí **0 = no aprende**. fixed_goal despega; random_goal se queda en el piso.

![Corridas prob](assets/prob_runs.png)

## Reproducibilidad

- Los comandos exactos de cada corrida están en
  [`../../execution_commands.md`](../../execution_commands.md) (numerados).
- Cada corrida guarda su config compuesta en `<logdir>/.hydra/config.yaml`:
  **ese** es el registro de verdad de sus hiperparámetros, no los defaults
  actuales.
- Las figuras de diagnóstico (`experiments/`) se regeneran con los scripts de
  `experiments/` sobre el checkpoint correspondiente; ver
  [`diagnosticos_espacio_representacion.md`](diagnosticos_espacio_representacion.md).

## Qué falta por ejecutar y cómo atacar `random_goal`

El gap fixed↔random **no es una sola falla**: son **dos mecanismos distintos**,
uno por cada receta que se quedó pegada. Mezclarlos lleva a conclusiones
contradictorias, así que se tratan por separado.

> Dato base (vale para los dos casos): tanto el **goal que ve la política**
> (`dreamer.py:355`) como la **reward** (`rewards.py`) son **`z` puro** — `deter`
> nunca entra al goal. Y la posición del verde se decodifica **~0.91 desde
> `deter`** vs solo **~0.25 desde `z`**
> ([random_goal_vs_fixed_goal](random_goal_vs_fixed_goal.md)): `z` lleva **poca**
> info de la posición, pero **no cero** (~16 slots se mueven con la celda).

### Caso 1 — `full` + goal del buffer: el match exacto es demasiado estricto

`full` exige que **los 32 grupos coincidan a la vez** (todo-o-nada). Con un goal
del buffer de **otra** celda, los ~16 slots que sí codifican posición nunca
matchean → reward -1 siempre → -1001. Lo contraintuitivo: aquí **menos** info de
posición en `z` *ayudaría* — si `z` fuera de verdad ciego a la celda, el goal
cruzado matchearía y random funcionaría como fixed. El obstáculo es que `z` **no
es lo bastante ciego** para un match exacto; "poca info" todavía rompe un 32/32.

- **Salida natural:** un goal de la **misma** celda (imaginación, misma posición)
  o aflojar el "32 a la vez" (`argmax_full`/modas, o reward densa).

### Caso 2 — `prob` + imagination: la señal densa es demasiado chata

Aquí el goal viene del **mismo episodio** (misma celda) → es alcanzable por
construcción y **la posición no es el obstáculo** (el propio
[random_goal_vs_fixed_goal](random_goal_vs_fixed_goal.md) lo dice). El sospechoso
es otro: `prob = log_prob(goal).exp()` solo es **alto** si la distribución de `z`
del estado está **concentrada** alrededor del goal. Si el latente del WM es
**difuso** (alta entropía), entonces *incluso parado en el estado-goal* la
probabilidad del one-hot exacto es baja → reward **chata y ~0 en todas partes** →
sin gradiente → se clava en 0. Y random_goal, al mover el verde entre episodios,
da un WM más estocástico/difuso (ver [`analisis_trayectorias_crafter.md`](analisis_trayectorias_crafter.md):
`z` más difuso en ambientes variables, más picudo en fixed). **No contradice el
Caso 1:** lo que rompe aquí es la *difusez del latente*, no la posición — y encaja
con la sospecha de que el WM de random esté **sub-entrenado** sobre la variedad de
celdas.

### Diagnósticos baratos (correr **antes** de quemar runs de 500k)

0. **¿La `prob` es siquiera ganable?** (zanja el Caso 2) Sobre el WM de random vs
   fixed, medir (a) la **entropía** del prior/posterior `z` y (b) la **`prob` máx
   alcanzable**: `log_prob(goal).exp()` evaluada *en el propio estado-goal
   imaginado*. Si en random eso ya es ~0 estando en el goal → la reward es
   **inganable** y es problema de **nitidez del WM** (más/mejor entrenamiento,
   latente más picudo). Si la `prob` máx es alta pero el agente no llega → es
   problema de **política/crédito**, no del WM. Hoy ambas causas están confundidas.
1. **Probe `z`→posición vs `(z++deter)`→posición** sobre el WM de random (ya
   existe para fixed). Cuantifica cuánto se gana al meter `deter` en el goal.

### Recetas para `random_goal` **sin "trampa"**

"Trampa" = darle la `(x,y)` del verde, la reward real del env, o un goal hecho a
mano. Sin trampa = quedarse en el marco goal-latente.

| | Receta | Ataca | Por qué podría funcionar / matiz |
|---|---|---|---|
| **A** | **WM más nítido en random** (entrenar más, o latente más picudo: temperatura, balance de KL, menos grupos) | Caso 2 | Si la `prob` máx sube, la señal densa deja de ser chata. Es la lectura directa de la sospecha "falta entrenar el WM" |
| **B** | **Comparar modas, no muestras** (`argmax_full`, o `prob` sobre el modo) | Caso 1 y 2 | Elimina la varianza de Gumbel (la columna C de `goal_reachability` daba 0% aun para el mismo estado); barato y ya codeado. `argmax_full`+imagination en random **no se ha corrido** |
| **C** | **Horizonte de imaginación corto** (item 21, `goal_imag_horizon=5/8`) | Caso 2 | Goals más cercanos → estado-goal más alcanzable y distribución más concentrada → `prob` menos chata. Solo se probó h=15 |
| **D** | **Meter `deter` en el *conditioning* del goal** (no en el match de `full`) | especificación | Le da a la política capacidad de identificar el estado-goal. Ojo: **no** arregla la difusez de `prob`, y meter `deter` en el *match* de `full` empeoraría el caso cruzado — útil solo como conditioning |

Apuesta: el diagnóstico **0** decide casi todo. Si el Caso 2 es "WM difuso" →
receta **A**/**C**; si la `prob` ya era ganable → el problema es la política y
miramos **D**/credit assignment.

### Cola de ejecución pendiente

> **Actualización (jul 2026).** El item 28 (A/B del artefacto `train/rew=-1`) ya
> corrió: ver [artefacto_reward_exacto_gpu](artefacto_reward_exacto_gpu.md). La
> receta **B** (comparar modas) quedó implementada de facto en **todos** los
> rewards de match exacto por el fix `025f3b3`, y `row_by_row`+imagination
> **aprende en random_goal** — el Caso 1 se destraba (replicado en 3/3 seeds,
> plateau ≈ -290 a 1M; ni la réplica ni el doble de pasos eran el límite). El
> Caso 2 (`prob` chata) se **endurece**: los runs randomstart+HER de `prob`
> (seeds 1-2, jul 7) descartan a la vez "le faltaba HER" y "le faltaba un WM
> mejor entrenado" — sobre el mismo WM, mismo goal y misma seed, `row_by_row`
> aprende y `prob` no. La evidencia apunta a que `prob` es inganable en
> random_goal; el diagnóstico 0 lo cerraría formalmente.

| Pendiente | Origen | Nota |
|---|---|---|
| **`goal_sample=image` + `row_by_row`** en random_goal | [artefacto](artefacto_reward_exacto_gpu.md) | conecta el goal latente con la tarea del verde; **la palanca con más retorno esperado** (la receta actual ya saturó en ≈-290) |
| Ver `eval_video.mp4` de `/03` | [artefacto](artefacto_reward_exacto_gpu.md) | confirmar que repite el comportamiento del `/02` (persigue el `z`, ignora el verde) |
| Confirmar en barto qué `rewards.py` corrió el A/B | [artefacto](artefacto_reward_exacto_gpu.md) | cierra (o no) la causa `torch.compile` |
| Diagnóstico **0**: entropía `z` y `prob`-máx, random vs fixed | propuesto | cerraría formalmente que `prob` es inganable (HER y WM ampliado ya descartados) |
| `argmax_full` + imagination en **random_goal** | propuesto (receta **B**) | menos urgente: rowbyrow ya prueba el mecanismo |
| ~~`row_by_row` sobre el WM sin `h`~~ ✅ **hecho** (jul 26): sin `h` **mejora** (ma25 -180 vs -290) | [posterior_sin_deter](posterior_sin_deter.md) | falta replicarlo en más seeds (hoy 1 seed vs 3 con `h`) y ver el `eval_video` |
| Sweep de `goal_imag_horizon` en random (h=5/8/30) | `execution_commands` item 21 | sin correr; receta **C** |
| `argmax_full` desde WM-only en fixed_goal | `execution_commands` item 15 (corregido) | sin reportar |
| Rehacer **Fase 2** (text encoder) en **fixed_goal** con reward alcanzable | flag en [fase2](fase2_goal_desde_texto.md) | aísla el encoder sin el confound de random+`full` |
| **Post-train sin congelar el WM** | flag en [fase3](fase3_posttrain_wm_aleatorio.md) | mide el salto frente al WM congelado; útil como cota |
| **Joint desde 0** (sin post-train, WM+política juntos) | propuesto | controla si la separación WM→política ayuda o estorba vs entrenar end-to-end |
| Probe `z` vs `deter` (posición/inventario) | propuesto en [trayectorias](analisis_trayectorias_crafter.md) | diagnóstico barato (1) |
