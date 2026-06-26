# Papers sugeridos — literatura relacionada mapeada al proyecto

[← Índice](README.md)

Trabajos relevantes para los problemas abiertos de este proyecto (goal latente
discreto `z`, match exacto inalcanzable, `z` difuso, fuga de posición, gap
fixed↔random). La **lectura común de toda la literatura reciente**: se abandonó
el *match exacto en el latente* en favor de (a) latentes más **limpios /
task-relevant** y (b) goals como **distancias aprendidas** o **distribuciones con
incertidumbre explícita** (o *generados*), en vez de muestras one-hot comparadas
por igualdad — justo donde se rompe nuestro pipeline
([`hallazgo_goal_type_full`](hallazgo_goal_type_full.md)).

## Mapa rápido problema → papers

| Problema del proyecto | Papers |
|---|---|
| `full` / match exacto sobre latente muestreado (P≈4e-13) | Director, Contrastive-RL, RIG, HLPS, Diffusional Subgoals |
| `z` difuso / poca decisión en el latente ([trayectorias](analisis_trayectorias_crafter.md)) | DreamerV3, HRSSM, Task-Relevant Reconstruction |
| Posición se filtra a `z`/`deter` ([random vs fixed](random_goal_vs_fixed_goal.md)) | HRSSM, Task-Relevant Reconstruction |
| Recompensa densa `prob` demasiado chata ([recompensa_prob](recompensa_prob_funciona.md)) | LEXA, Contrastive-RL, RIG |
| Gap fixed↔random / señal de crédito que no llega | HIQL, LEXA, Diffusional Subgoals |
| Cómo obtener/generar goals latentes **alcanzables** | Director, RIG, HLPS, Diffusional Subgoals |

Las recetas a las que apuntan están en
[README → "Recetas para `random_goal` sin trampa"](README.md#recetas-para-random_goal-sin-trampa).

---

## Fundacionales (goals latentes + world models)

### Director — *Deep Hierarchical Planning from Pixels*
Hafner, Lee, Fischer, Abbeel — NeurIPS 2022 — [arXiv:2206.04114](https://arxiv.org/abs/2206.04114)

Política jerárquica **sobre el latente de un Dreamer**: un *manager* propone goals
latentes y un *worker* los alcanza. Clave para nuestro muro: en vez de muestrear
goals en el latente crudo, entrenan un **goal autoencoder** que comprime el estado
a códigos discretos *reconstruibles*, garantizando goals representables/alcanzables,
y la recompensa del worker es una **similitud (max-cosine)**, no un match exacto
one-hot. Es la salida directa a `goal_type=full` y a generar goals alcanzables.

### LEXA — *Discovering and Achieving Goals via World Models*
Mendonca, Rybkin, Daniilidis, Hafner, Pathak — NeurIPS 2021 — [arXiv:2110.09514](https://arxiv.org/abs/2110.09514)

Goal-conditioned **sobre world models** (explorer + achiever). Compara reward de
**distancia temporal** (cuántos pasos faltan, aprendido) vs **similitud latente**, y
el de distancia temporal generaliza mejor a goals lejanos. Ataca de frente el match
exacto (inalcanzable bajo Gumbel) y la `prob` chata: da señal densa **sin** depender
de que el latente sea picudo. Conecta con la receta C (horizontes cortos / goals
cercanos).

### Contrastive Learning as Goal-Conditioned RL
Eysenbach, Zhang, Salakhutdinov, Levine — NeurIPS 2022 — [arXiv:2206.07568](https://arxiv.org/abs/2206.07568)

Plantea que alcanzar goals *es* aprendizaje contrastivo: el reward de goal-reaching
emerge del producto interno entre estado-acción y goal futuro, sin reconstrucción ni
match exacto. Es el marco principista para reemplazar `full`/`argmax_full` por una
**métrica aprendida de alcanzabilidad**, robusta a un latente estocástico — la
pregunta de fondo de [`analisis_trayectorias_crafter`](analisis_trayectorias_crafter.md).

### RIG — *Visual Reinforcement Learning with Imagined Goals*
Nair, Pong, Dalal, Bahl, Lin, Levine — NeurIPS 2018 — [arXiv:1807.04742](https://arxiv.org/abs/1807.04742)

Antecedente clásico de "goal como latente" e **imaginar goals**: muestrea goals del
prior latente (nuestro `goal_sample=imagination` es la versión world-model) y define
reward como **distancia latente** `-‖z − z_goal‖`, con HER en ese espacio. Muestra
que una **distancia continua** funciona donde la igualdad discreta no, y subraya que
el espacio latente debe estar bien estructurado para que la distancia sea
significativa — nuestra tensión `z` difuso vs info en `h`.

### DreamerV3 — *Mastering Diverse Domains through World Models*
Hafner, Pasukonis, Ba, Lillicrap — 2023 — [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

No es goal-conditioned, pero es la referencia para el **`z` difuso**. Mismo latente
categórico (32×32) y varias palancas para *controlar la nitidez/estabilidad* del
posterior: **KL balancing**, **free bits**, mezcla con uniforme. Son las palancas de
la receta A (WM más nítido) que nuestro Barlow Twins **no** aplica, y la base para el
diagnóstico 0 (entropía del prior/posterior).

---

## Recientes (2024-2025)

### HRSSM — *Learning Latent Dynamic Robust Representations for World Models*
Sun, Zang, Li, Islam — ICML 2024 — [arXiv:2405.06263](https://arxiv.org/abs/2405.06263)

Ataca directamente "el `z` arrastra info irrelevante (posición, ruido)". Parte de que
Dreamer sufre con ruido exógeno e irrelevante y propone un **Hybrid RSSM** con
*masking espacio-temporal* + **bisimulación** + reconstrucción latente, para que el
estado capture solo lo **endógeno/task-relevant** y filtre correlaciones espurias. En
vez de *medir* cuánta info útil hay en `z` vs `h`, da un objetivo que *empuja* el
latente a ser limpio (receta A / problema de fuga de posición).

### Make the Pertinent Salient — *Task-Relevant Reconstruction for Visual Control with Distractions*
2024 — [arXiv:2410.09972](https://arxiv.org/abs/2410.09972)

Observa que reconstruir la imagen **completa** empuja al encoder a retener *toda* la
información sin importar su relevancia, desperdiciando capacidad — la versión
"con decoder" de nuestro `z` difuso. Propone reconstrucción **selectiva** de lo
relevante a la tarea. Útil como contraste de diseño frente a nuestro objetivo
sin-decoder (Barlow no premia decisión, problema 2).

### HLPS — *Probabilistic Subgoal Representations for Hierarchical RL*
Wang, Wang, Yang, Kämäräinen, Pajarinen — ICML 2024 — [arXiv:2406.16707](https://arxiv.org/abs/2406.16707)

El más alineado con el muro de `goal_type=full`. En lugar de un mapeo determinista
estado→subgoal (o nuestro sample==sample), modela el subgoal como una **distribución**
vía Procesos Gaussianos, tratando la representación latente como **observación
corrupta por ruido** (`f = z + ε`, varianza *aprendible*) — captura estocasticidad
ambiental y regiones poco exploradas por la varianza posterior. Es el marco que
nuestros diagnósticos piden: alcanzar/comparar goals **bajo incertidumbre explícita**,
no por igualdad de one-hots.

### Hierarchical RL with Uncertainty-Guided Diffusional Subgoals
Wang, Wang, Pajarinen — 2025 — [arXiv:2505.21750](https://arxiv.org/abs/2505.21750)

Evolución del anterior, on-theme para "cómo generar goals latentes alcanzables". Usa
un **modelo de difusión** para *generar* candidatos de subgoal latente y una
**selección guiada por incertidumbre** para quedarse con los factibles/informativos.
Alternativa principista a nuestro `goal_sample=imagination` (rollout aleatorio): en
vez de tomar el `z` prior final de acciones random, *aprendes* a generar goals
factibles y útiles — apuntando al caso random_goal.

### HIQL — *Offline Goal-Conditioned RL with Latent States as Actions*
Park, Ghosh, Eysenbach, Levine — NeurIPS 2023 — [arXiv:2307.11949](https://arxiv.org/abs/2307.11949)

Fundacional para nuestro setup: la política de alto nivel **trata estados como
acciones** y predice una *representación latente de un subgoal*; la de bajo nivel lo
alcanza. Clave: está diseñado para ser **robusto al ruido en la value function** — en
vez de estimar el valor de goals lejanos (ruidoso e inestable, nuestro síntoma
`loss/policy`≈0), descompone en subgoals cercanos donde la señal es fiable. Conecta el
gap fixed↔random con una jerarquía que reduce la dependencia de estimaciones ruidosas
(receta C).

---

## Cómo se traduce a acciones concretas

Estas lecturas motivan, en orden de leverage/costo (detalle de la discusión en la
bitácora):

1. **Reemplazar el match exacto por similitud/distancia** sobre `get_feat`
   (`z++deter`), no sobre muestras de `z` — un `goal_type` nuevo en `rewards.py`.
   *(Director, RIG)*
2. **Distancia aprendida** (contrastiva / temporal) si la señal a mano no alcanza.
   *(LEXA, Contrastive-RL, HIQL)*
3. **Latente más nítido / task-relevant**: KL balancing, free bits, temperatura
   *(DreamerV3)*; masking/bisimulación *(HRSSM, Task-Relevant Reconstruction)*.
4. **Generar goals alcanzables** con goal autoencoder o difusión + incertidumbre.
   *(Director, Diffusional Subgoals, HLPS)*
