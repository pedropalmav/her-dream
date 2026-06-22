# Hallazgo central — `goal_type=full` es inalcanzable por construcción (2026-06-02)

[← Índice](README.md)

Este es el resultado que explica **todas** las fallas a -1001 del proyecto. No es
el text encoder, no es el WM, no es HER, no es el tamaño del RSSM: es la **forma
de la recompensa**.

## Qué es `goal_type=full`

`full_goal_reward` da recompensa **0 si y sólo si los 32 grupos one-hot de `z`
coinciden exactamente** entre una muestra del estado y una muestra del goal, y
**-1 en cualquier otro caso**. Compara `sample == sample`: dos muestras one-hot
*independientes* del Gumbel-Softmax.

## El experimento `text_wm_alignment.py`

Sobre `distill_text_from_wm_only/01` (120 trayectorias, 1200 pares, 94 misiones
únicas) se empareja, sobre rollouts **reales**, cada `obs["mission"]` con el
posterior del WM `q_wm(z|obs)` del **mismo paso** y se compara contra
`q_text(z|mission)`.

### Conclusión 1 — la destilación SÍ funciona (no es el problema)

Todo está muy por encima del azar (1/K = 0.0625):

| Métrica | Valor |
|---|---|
| Colisión media por slot ⟨p_t, p_w⟩ | **0.415** |
| Acuerdo argmax medio | 0.58 |
| Colisión slot 0 | 0.391 |
| Text dentro del soporte del WM | 100% |

Matiz: el WM es casi determinista en el slot 0 (entropía H ≈ 0.11) mientras el
text encoder es **difuso** (H ≈ 1.41 de 2.77 máx) y colapsa las 94 misiones
sobre **7 de 16 clases**. El goal de texto identifica *pobremente* la misión,
pero el encoder **no está roto ni al azar**.

### Conclusión 2 — el bloqueante es `goal_type=full`

Con colisión media 0.42 por slot, la probabilidad de que coincidan los **32**
slots a la vez es el producto:

```
P(match completo, 32 slots) = ∏_s colisión_s ≈ 3.8e-13   (log10 ≈ -12.4)
```

Astronómicamente pequeña → **el reward es esencialmente siempre -1** → la
política nunca ve señal positiva → no hay gradiente → no aprende. Explica el
síntoma exacto: `score` -1001, `loss/policy` ≈ -5e-4.

> **Nota técnica.** El propio `text_wm_alignment.py` está escrito asumiendo
> `goal_type=first_row` (analiza sólo el slot 0). Para estos checkpoints esa
> premisa **subestima** el problema: el reward real es el producto sobre los 32
> slots, no sólo el slot 0.

## Por qué Fase 1 sí funcionaba

Porque **Fase 1 no usaba `full`**: usaba el reward de **primera fila**
(`stoch[:, 0]`, un solo grupo), que es anterior a que existiera `goal_type`
(verificado contra el `.hydra/config.yaml`; ver
[fase1](fase1_primera_row_alcanzable.md)). Con un solo grupo el match es alcanzable
(colisión ~0.39 en el slot 0), así que la política recibe señal y aprende
(-357/-417), **incluso en `random_goal`**. El salto a -1001 ocurre exactamente
cuando se pasa a exigir los **32 grupos a la vez** (`full`): el producto de
colisiones por slot colapsa a ~0. No cambió el entorno ni la fuente del goal;
cambió de 1 fila a 32 filas.

## Las salidas (implementadas en código, 03–05 jun)

El diagnóstico señaló dos direcciones, ambas ya en `rewards.py`:

1. **Comparar modas en vez de muestras** — `goal_type=argmax_full` (PR #25,
   `4d07ef8`): compara el goal contra el **argmax one-hot de los prior logits**,
   eliminando la varianza de Gumbel del lado del estado. Sube el match efectivo
   del slot 0 a ~0.55.
2. **Aflojar el "32 a la vez"** — `goal_type=first_row` / `row_by_row`: el reward
   sobre el slot 0 esperaba señal ~0.39 (vs 4e-13 con `full`).
3. **Relajación blanda del match** — `goal_type=log_prob` (PR #26): reward 0 si
   `dist.log_prob(goal) >= log(prob_threshold)` — premia que el goal sea
   *probable* bajo el prior, no idéntico a una muestra.

La que termina **funcionando de verdad** es la versión densa de (3),
`goal_type=prob` (sin umbral, reward = `log_prob(goal).exp()` ∈ [0,1]) sobre un
goal alcanzable de imaginación → [recompensa_prob_funciona](recompensa_prob_funciona.md).

## Tabla de `goal_type` (de `rewards.py`)

| `goal_type` | Goal | Reward |
|---|---|---|
| `first_row` | `(K,)` | 0 si slot 0 coincide, else -1 |
| `row_by_row` | `(S,K)` | `(filas que matchean / S) − 1` — densa en [-1,0] |
| `full` | `(S,K)` | 0 si **los S grupos** matchean exacto, else -1 ← **inviable** |
| `argmax_full` | `(S,K)` | como `full` pero contra el argmax one-hot del prior |
| `log_prob` | `(S,K)` | 0 si `log_prob(goal) ≥ log(thr)`, else -1 |
| `prob` | `(S,K)` | `log_prob(goal).exp()` ∈ [0,1] — **densa, sin umbral** ← funciona |

## Artefactos

`logdir/distill_text_from_wm_only/01/experiments/text_wm_align/`:
`align_agreement_per_slot.png`, `align_slot0_entropy.png`,
`align_slot0_confusion.png`, `align_slot0_coverage.png`, `align_results.json`.
