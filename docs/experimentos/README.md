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
| [`fase1_el_setup_funciona.md`](fase1_el_setup_funciona.md) | El setup base aprende cuando el goal sale del posterior del WM (seeds 3/4 → -357/-417). |
| [`fase2_goal_desde_texto.md`](fase2_goal_desde_texto.md) | Mover la fuente del goal al text-encoder rompe el aprendizaje (todo a -1001). |
| [`fase3_aislar_wm_vs_politica.md`](fase3_aislar_wm_vs_politica.md) | WM congelado + post-train: el WM no era el cuello de botella. |
| [`hallazgo_goal_type_full.md`](hallazgo_goal_type_full.md) | **El bloqueante central:** `goal_type=full` (sample==sample sobre 32 grupos) es inalcanzable por construcción (P≈4e-13). |
| [`random_goal_vs_fixed_goal.md`](random_goal_vs_fixed_goal.md) | Por qué post-train funciona en fixed_goal y falla en random_goal: la posición del verde vive en `z`/`deter`. Incluye `goal_sample=imagination`. |
| [`recompensa_prob_funciona.md`](recompensa_prob_funciona.md) | **El desenlace positivo:** recompensa densa `prob` + goal de imaginación → aprende en fixed_goal (+331) y en random_goal (0). |
| [`diagnosticos_espacio_representacion.md`](diagnosticos_espacio_representacion.md) | Las herramientas de `experiments/` (estocasticidad del WM/text, consistencia del posterior). |

## La historia en una línea

El reward `z(estado) == z(goal)` **es entrenable** (Fase 1), pero exigir match
**exacto de muestra-vs-muestra sobre los 32 grupos** (`goal_type=full`) lo vuelve
inalcanzable por el ruido de Gumbel del muestreo. Todas las fallas posteriores
(goal de texto, goal de buffer en random_goal, goal de imaginación) son síntomas
del mismo muro. La salida que **sí funciona** es cambiar la *forma* de la
recompensa: una densa por **probabilidad de sampleo** (`goal_type=prob`) sobre un
goal **alcanzable por construcción** (`goal_sample=imagination`).

## Tabla maestra de corridas

Score = media de los últimos 10 episodios (piso = **-1001**, el time-limit sin
alcanzar el goal; en `crafter` la escala es otra). 🟢 = aprende, 🔴 = pegado al piso.

| Corrida | Env | goal_sample | goal_type | buffer | Steps | Score final | Máx | |
|---|---|---|---|---|---|---|---|:--:|
| `goal_dreamer_with_text/03` (seed 3) | random | buffer (1ª fila) | first_row† | normal | 499k | **-357** | -7 | 🟢 |
| `goal_dreamer_with_text/04` (seed 4) | random | buffer (1ª fila) | first_row† | normal | 499k | **-417** | -9 | 🟢 |
| `goal_dreamer_with_text/01` (seed 0) | random | buffer (1ª fila) | first_row† | normal | 9k | -952 | -691 | ⏸ corta |
| `text_goal_sample_her_buffer/01` | fixed | text | full | HER | 499k | -1001 | -1001 | 🔴 |
| `text_goal_sample_normal_buffer/01` | fixed | text | full | normal | 499k | -1001 | -1001 | 🔴 |
| `text_goal_sample_8x8_goal_her_buffer/01` (RSSM 8×8) | fixed | text | full | HER | 499k | -998 | -121 | 🔴 |
| `text_goal_sample_8x8_normal_buffer/01` (RSSM 8×8) | fixed | text | full | normal | 499k | -1000 | -811 | 🔴 |
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
| **`post_train_from_wm_only/04_imag_prob/01`** | **fixed** | **imagination** | **prob** | normal | 499k | **+331** | +727 | 🟢 |
| **`random_goal/...frozenwm_normalbuf_goalimag_prob/01`** | **random** | **imagination** | **prob** | normal | 499k | **0** | +73 | 🟢 |
| `original_wm_crafter/02` (baseline vanilla) | crafter | — | — | — | 1.01M | eval≈10.9 | 14.1 | 🟢 WM juega |
| `z_without_history_wm_crafter/01` (ablación) | crafter | — | — | — | 1.01M | eval≈8.7 | 12.1 | 🟢 −13% |

> † Las corridas `goal_dreamer_with_text/*` (abr-28) son **anteriores** a que
> existieran los keys `goal_type`/`goal_sample`: corrieron con el reward de
> **primera fila** hardcodeado (`stoch[:, 0]`), no con `full`. El equivalente
> actual es `goal_type=first_row`. Verificado contra su `.hydra/config.yaml`.
> Detalle en [fase1](fase1_el_setup_funciona.md).

## Figuras de conjunto

Generadas por [`scripts/plot_runs.py`](scripts/plot_runs.py) a partir de los
eventos de TensorBoard. Re-ejecutar con `uv run python3
docs/experimentos/scripts/plot_runs.py` cuando lleguen corridas nuevas.

### Qué sirve y qué no — score final por corrida

![Score final por corrida](assets/score_final_barras.png)

### Curvas de entrenamiento por fase

![Curvas por fase](assets/curvas_por_fase.png)

## Reproducibilidad

- Los comandos exactos de cada corrida están en
  [`../../execution_commands.md`](../../execution_commands.md) (numerados).
- Cada corrida guarda su config compuesta en `<logdir>/.hydra/config.yaml`:
  **ese** es el registro de verdad de sus hiperparámetros, no los defaults
  actuales.
- Las figuras de diagnóstico (`experiments/`) se regeneran con los scripts de
  `experiments/` sobre el checkpoint correspondiente; ver
  [`diagnosticos_espacio_representacion.md`](diagnosticos_espacio_representacion.md).
