1. Correr modelo
```bash
ts -G 1 bash scripts/train.sh logdir=./logdir/full_goal_dreamer_with_text/01 seed=1 mission_text=True trainer.steps=100000 trainer.update_log_every=1000
```

1.b Correr modelo en la segunda GPU (GPU 1) — usa `goal_sample=text` para muestrear goals desde el text encoder vivo:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh logdir=./logdir/wm_only_random_mission/01 env=fixed_goal seed=1 mission_text=True env.goal_sample=random trainer.steps=500000 buffer=normal wm_only=True
```

2. Traerse el tensorboard
```bash
scp -r iamonardes@barto.ing.uc.cl:/home/iamonardes/her-dream/logdir/random_goal/posttrain_randomstart_goalimag_rowbyrow/03 ./logdir/random_goal/posttrain_randomstart_goalimag_rowbyrow/03
```

3. Correr el tensorboard
```bash
tensorboard --logdir ./logdir/random_goal/posttrain_randomstart_goalimag_rowbyrow_fixedrew/01
```

4. Correr evaluación con el text_encoder al azar:
```bash
python3 eval_text_goal.py --logdir logdir/goal_dreamer_with_text/04 --episodes 10 --device cpu
```


4.b Post-training: cargar el WM congelado desde `wm_only_random_mission/01` y entrenar solo actor/critic (puedes cambiar logdir, goal_sample, buffer, steps, etc.):
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/wm_only_random_mission/01 \
    logdir=./logdir/post_train_from_wm_only_sample_buffer/01 \
    freeze_wm=True wm_only=False \
    env=fixed_goal mission_text=True env.goal_sample=buffer \
    buffer=normal seed=1 trainer.steps=500000 trainer.update_log_every=1000
```

5. Correr codigo de pedro:
```bash
ts -G 1 bash scripts/train.sh logdir=./logdir/observed-z-goals/01 seed=1 trainer.steps=1000000 trainer.update_log_every=1000 env.goal_sample=buffer buffer=her env=fixed_goal
```


6. Ver estado actual de la ejecucion:
```bash
ts -t {process_id}
```

7. Ver todo el estado de la ejecucion:
```bash
ts -c {process_id}
```

8. Ver todos los procesos propios:
```bash
ts
```

9. Matar proceso:
```bash
ts -k {process_id}
```

10. Post train:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
      load_from=./logdir/wm_only_random_mission/01 \
      logdir=./logdir/post_train_from_wm_only/01 \
      freeze_wm=True wm_only=False \
      env=fixed_goal mission_text=True env.goal_sample=random \
      buffer=normal seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

11. Post train con goals sampleados del replay buffer (misiones pasadas), sin HER:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
      load_from=./logdir/wm_only_random_mission/01 \
      logdir=./logdir/post_train_from_wm_only/02_normal_buffer_goals \
      freeze_wm=True wm_only=False \
      env=fixed_goal mission_text=True env.goal_sample=buffer \
      buffer=normal seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

12. Post train con goals sampleados del replay buffer (misiones pasadas), con HER:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
      load_from=./logdir/wm_only_random_mission/01 \
      logdir=./logdir/post_train_from_wm_only/03_her_buffer_goals \
      freeze_wm=True wm_only=False \
      env=fixed_goal mission_text=True env.goal_sample=buffer \
      buffer=her seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

13. Fase B — Destilar el text encoder sobre el WM congelado. Carga el checkpoint de `wm_only_random_mission/01`, mantiene WM y actor/critic fijos y entrena SOLO el `TextEncoderGRU` contra el posterior congelado (target estacionario). Usa el mismo `env`/`buffer` que la corrida wm_only (item 1.b) para que la distribución de datos coincida; `goal_sample=random` evita usar el text encoder (aún sin entrenar) para muestrear goals. `text_kl=1.0` porque es la única loss:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh train_text_only=True \
      load_from=./logdir/wm_only_random_mission/01 \
      logdir=./logdir/distill_text_from_wm_only/01 \
      mission_text=True \
      env=fixed_goal env.goal_sample=random buffer=normal \
      seed=1 trainer.steps=200000 trainer.update_log_every=1000 \
      model.loss_scales.text_kl=1.0
```

14. Fase C — Entrenar la política sobre el checkpoint destilado (item 13), con WM y text encoder ya entrenados y AMBOS congelados. `goal_sample=text` muestrea los goals desde el text encoder entrenado en la fase B:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
      load_from=./logdir/distill_text_from_wm_only/01 \
      logdir=./logdir/post_train_from_distill/01 \
      freeze_wm=True wm_only=False mission_text=True \
      env=fixed_goal env.goal_sample=text \
      buffer=her seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

15. Igual que el item 14 (post-train sobre el checkpoint **destilado**, item 13) pero con reward **argmax_full** (doble argmax: moda del estado imaginado vs moda del goal). Goals desde el **text encoder** ya entrenado en la fase B (`goal_sample=text`); con `goal_type=argmax_full` el goal de texto se toma como `argmax(text_logits)`. Reward argmax_full + HER, sobre `fixed_goal`:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
      load_from=./logdir/distill_text_from_wm_only/01 \
      logdir=./logdir/post_train_from_distill/02_argmax_text_her \
      freeze_wm=True wm_only=False mission_text=True \
      env=fixed_goal env.goal_sample=text \
      buffer=her goal_type=argmax_full seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

16. Post-train directo desde el **WM-only** `wm_only_random_mission/01` (item 1.b, **no** desde el destilado), con goals muestreados desde el **replay buffer** (`goal_sample=buffer`); con `goal_type=argmax_full` el goal se toma como `argmax(data["logit"])`. Reward argmax_full + HER, sobre `fixed_goal`:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
      load_from=./logdir/wm_only_random_mission/01 \
      logdir=./logdir/post_train_from_wm_only/06_argmax_buffer_her \
      freeze_wm=True wm_only=False mission_text=True \
      env=fixed_goal env.goal_sample=buffer \
      buffer=her goal_type=argmax_full seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

17. **random_goal — Fase 1: WM only.** Pre-entrena solo el world model en el ambiente `random_goal` (cuadrado verde se mueve al terminar el episodio), sin texto. Base para los post-trains de los items 18-19, para poder comparar de forma justa contra el run joint-from-scratch (`random_goal_herbuf_goalbuf`). `env.steps` se usa porque `trainer.steps` lo hereda (`trainer.steps: ${env.steps}`):
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    logdir=./logdir/random_goal/wm_only_randomgoal/01 \
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=random buffer=normal wm_only=True \
    env.steps=500000 trainer.update_log_every=1000
```

18. **random_goal — Fase 2a: Post-train con WM congelado, HER + goals del buffer.** Réplica del run fallido pero partiendo del WM pre-entrenado/congelado del item 17 (esperar a que termine). Solo entrena actor/critic:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomgoal/01 \
    logdir=./logdir/random_goal/posttrain_frozenwm_herbuf_goalbuf/01 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=buffer buffer=her \
    env.steps=500000 trainer.update_log_every=1000
```

19. **random_goal — Fase 2b: Post-train con WM congelado, buffer normal + goals del buffer (sin HER).** Baseline para aislar el efecto de HER frente al item 18:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomgoal/01 \
    logdir=./logdir/random_goal/posttrain_frozenwm_normalbuf_goalbuf/01 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=buffer buffer=normal \
    env.steps=500000 trainer.update_log_every=1000
```

20. **random_goal — Fase 2c: Post-train con WM congelado y `goal_sample=imagination`.** Igual base que los items 18-19 (WM pre-entrenado/congelado del item 17), pero el goal de cada episodio se genera imaginando `imag_horizon` pasos con acciones aleatorias desde la primera observación (`Dreamer.imagine_goal`), así es alcanzable por construcción dentro del mismo episodio — la hipótesis para destrabar `random_goal` frente a los goals del buffer. `buffer=normal` (sin HER, ya que el goal imaginado no se re-etiqueta):
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomgoal/01 \
    logdir=./logdir/random_goal/posttrain_frozenwm_normalbuf_goalimag/01 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=imagination buffer=normal \
    env.steps=500000 trainer.update_log_every=1000
```

 21. **random_goal — Fase 2c con horizonte de goal imaginado configurable.** Igual que el item 20, pero usando `model.goal_imag_horizon` para controlar cuántos pasos se imaginan al generar el goal (antes era fijo en `imag_horizon=15`). Un horizonte más corto da goals más cercanos/alcanzables; uno más largo, goals más lejanos. Default = `imag_horizon` si se omite. El `logdir` incluye el horizonte para no pisar otras corridas:                                                
```bash                                                                         
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomgoal/01 \
    logdir=./logdir/random_goal/posttrain_frozenwm_normalbuf_goalimag_h30/01 \
    freeze_wm=True wm_only=False \ 
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=imagination buffer=normal model.goal_imag_horizon=30 \
    env.steps=500000 trainer.update_log_every=1000
```

22. **fixed_goal — Post-train con WM congelado, `goal_sample=imagination` y recompensa `prob`.** Goal imaginado (alcanzable por construcción) + recompensa densa por probabilidad de sampleo: `dist.log_prob(goal).exp()` ∈ [0,1], la probabilidad de samplear el goal bajo la distribución del estado imaginado (sin umbral). Parte del WM pre-entrenado/congelado de `wm_only_random_mission/01` (item 1.b); `mission_text=True` para casar con ese checkpoint. `buffer=normal` (el goal imaginado no se re-etiqueta, sin HER):
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/post_train.sh \
    load_from=./logdir/wm_only_random_mission/01 \
    logdir=./logdir/post_train_from_wm_only/04_imag_prob/01 \
    freeze_wm=True wm_only=False mission_text=True \
    env=fixed_goal env.goal_sample=imagination \
    buffer=normal goal_type=prob seed=1 \
    trainer.steps=500000 trainer.update_log_every=1000
```

23. **random_goal — Post-train con WM congelado, `goal_sample=imagination` y recompensa `prob`.** Igual que el item 22 pero en `random_goal`, partiendo del WM congelado de `wm_only_randomgoal/01` (item 17), sin texto. Compara directamente contra el item 20 (mismo goal imaginado pero recompensa `full`), aislando el efecto de usar la probabilidad de sampleo como recompensa densa:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/post_train.sh \
    load_from=./logdir/random_goal/wm_only_randomgoal/01 \
    logdir=./logdir/random_goal/posttrain_frozenwm_normalbuf_goalimag_prob/01 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=imagination buffer=normal goal_type=prob \
    env.steps=500000 trainer.update_log_every=1000
```

24. **fixed_goal — igual que item 22 pero con HER (`buffer=her`).** Versión con re-etiquetado de goals del item 22 (`goal_sample=imagination` + `goal_type=prob`, WM congelado de `wm_only_random_mission/01`). Aísla si HER ayuda a la recompensa densa `prob` sobre el goal imaginado:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/post_train.sh \
    load_from=./logdir/wm_only_random_mission/01 \
    logdir=./logdir/post_train_from_wm_only/05_imag_prob_her/01 \
    freeze_wm=True wm_only=False mission_text=True \
    env=fixed_goal env.goal_sample=imagination \
    buffer=her goal_type=prob seed=1 \
    trainer.steps=500000 trainer.update_log_every=1000
```

25. **random_goal — igual que item 23 pero con HER (`buffer=her`).** Versión con HER del item 23, partiendo del WM congelado de `wm_only_randomgoal/01` (item 17), sin texto. Es la prueba directa de si el re-etiquetado HER rescata el caso `random_goal` (que con `buffer=normal` se queda en el piso 0 de la recompensa `prob`, sin aprender):
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/post_train.sh \
    load_from=./logdir/random_goal/wm_only_randomgoal/01 \
    logdir=./logdir/random_goal/posttrain_frozenwm_herbuf_goalimag_prob/01 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=imagination buffer=her goal_type=prob \
    env.steps=500000 trainer.update_log_every=1000
```

26. **fixed_goal — Joint desde 0 (WM + política juntos), espejo del item 22.** Mismo recipe que el item 22 (`goal_sample=imagination` + `goal_type=prob`, `fixed_goal`, `mission_text=True`) pero **end-to-end** y con **HER** (`buffer=her`): se omiten `load_from`, `freeze_wm` y `wm_only`, así el world model y el actor/critic se entrenan juntos desde cero. Es el control directo del post-train con WM congelado: si el joint aprende y el post-train no (o viceversa), aísla si el cuello de botella es el WM o la política:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    logdir=./logdir/joint_from_scratch/01_imag_prob_her \
    mission_text=True \
    env=fixed_goal env.goal_sample=imagination \
    buffer=her goal_type=prob seed=1 \
    trainer.steps=500000 trainer.update_log_every=1000
```
   - Para la variante **random_goal**: `env=random_goal mission_text=False` y usar `env.steps=500000` en lugar de `trainer.steps` (lo hereda vía `trainer.steps: ${env.steps}`).
   - Para la variante **full + goals del buffer**: `env.goal_sample=buffer goal_type=full` (quita `goal_sample=imagination`/`goal_type=prob`); con goals del buffer se puede sumar `buffer=her` para re-etiquetar.

27. **random_goal — WM only con inicio del agente aleatorio, 1M steps.** Igual que el item 17 (pre-entrena solo el world model en `random_goal`, sin texto, acciones aleatorias) pero ahora el **agente** también parte en una celda interior uniformemente aleatoria cada episodio (`env.agent_start_random=True`, además del cuadrado verde que ya se movía), y entrenado por **el doble de pasos (1M)**. Amplía la distribución del WM para que tanto el agente como el goal cubran toda la grilla. `logdir` nuevo para no pisar el WM del item 17. `env.steps` se usa porque `trainer.steps` lo hereda (`trainer.steps: ${env.steps}`):
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    logdir=./logdir/random_goal/wm_only_randomstart/01 \
    env=random_goal seed=1 mission_text=False \
    env.goal_sample=random env.agent_start_random=True \
    buffer=normal wm_only=True \
    env.steps=1000000 trainer.update_log_every=1000
```

Luego, post-entrenar la **política** sobre este WM (congelado), manteniendo el mismo ambiente (`env=random_goal env.agent_start_random=True`) para que la distribución calce. Goal por imaginación (alcanzable por construcción) + `buffer=her`. Dos comandos para comparar full vs prob sobre el mismo WM ampliado:

Reward **full** (match exacto de los 32 grupos):
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomstart/01 \
    logdir=./logdir/random_goal/posttrain_randomstart_goalimag_full/01 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=1 mission_text=False \
    env.agent_start_random=True env.goal_sample=imagination \
    buffer=her goal_type=full \
    env.steps=500000 trainer.update_log_every=1000
```

Reward **prob** (densa, `dist.log_prob(goal).exp()` ∈ [0,1]); idéntico salvo `goal_type`:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomstart/01 \
    logdir=./logdir/random_goal/posttrain_randomstart_goalimag_prob/02 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=2 mission_text=False \
    env.agent_start_random=True env.goal_sample=imagination \
    buffer=her goal_type=prob \
    env.steps=500000 trainer.update_log_every=1000
```

Reward **row_by_row** (densa, `(filas que matchean / S) - 1` ∈ [-1, 0]); reutiliza el mismo WM congelado que `full`/`prob`, idéntico salvo `goal_type` y `logdir`:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomstart/01 \
    logdir=./logdir/random_goal/posttrain_randomstart_goalimag_rowbyrow/03 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=3 mission_text=False \
    env.agent_start_random=True env.goal_sample=imagination \
    buffer=her goal_type=row_by_row \
    env.steps=1000000 trainer.update_log_every=1000
```

28. **Diagnóstico — ¿por qué `train/rew ≈ -1` en los goal_type de match exacto (`full`/`row_by_row`)?** El item 27 (`row_by_row`) logueó `train/rew ≈ -1.0` (0 filas calzadas en TODO el batch de imaginación) pese a que `episode/score` era denso (~-650 ≈ 11 filas). Reproducir el camino de `_cal_grad` fiel en CPU da ~-0.65, **no** -1. La única diferencia con el run real es el paso de entrenamiento en GPU: `fp16 autocast` (descartado como causa: ni CPU ni MPS castean el one-hot del gumbel) + `torch.compile(reduce-overhead/cudagraphs)` (sospechoso principal, no reproducible sin CUDA). Estos 3 runs cortos aíslan la causa y prueban el fix. Runs de ~20k pasos bastan: el `-1` aparece desde la primera actualización. Ver memoria `exact-match-reward-flat-minus1-gpu`.

   **A) Diagnóstico — recompensa `==` actual + `compile=TRUE`** (reproduce el bug). Usa el `rewards.py` que está hoy en barto (comparación float `==`):
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomstart/01 \
    logdir=./logdir/dbg/rbr_eq_compile/01 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=1 mission_text=False \
    env.agent_start_random=True env.goal_sample=imagination \
    buffer=her goal_type=row_by_row model.compile=True \
    env.steps=20000 trainer.update_log_every=200
```

   **B) Diagnóstico — recompensa `==` actual + `compile=FALSE`** (idéntico a A salvo `model.compile`). Aísla `torch.compile` como única variable:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomstart/01 \
    logdir=./logdir/dbg/rbr_eq_nocompile/01 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=1 mission_text=False \
    env.agent_start_random=True env.goal_sample=imagination \
    buffer=her goal_type=row_by_row model.compile=False \
    env.steps=20000 trainer.update_log_every=200
```

   **C) Fix — recompensa argmax (nueva) + `compile=TRUE` + assert de one-hot activo.** Requiere aplicar el fix de `rewards.py` (compara `argmax`, no `==`). `VALIDATE_ONEHOT=1` corre el assert que valida que `imag_stoch` sea un one-hot legítimo (moda a <1e-3 de 1.0):
```bash
CUDA_VISIBLE_DEVICES=1 VALIDATE_ONEHOT=1 ts -G 1 bash scripts/train.sh \
    load_from=./logdir/random_goal/wm_only_randomstart/01 \
    logdir=./logdir/dbg/rbr_argmax_compile/01 \
    freeze_wm=True wm_only=False \
    env=random_goal seed=1 mission_text=False \
    env.agent_start_random=True env.goal_sample=imagination \
    buffer=her goal_type=row_by_row model.compile=True \
    env.steps=20000 trainer.update_log_every=200
```

   **Cómo leer los resultados** (mirar `train/rew` en TensorBoard o `metrics.jsonl`):
   - **A ≈ -1 y B ≈ -0.65** ⟹ confirmado: la causa es `torch.compile`/cudagraphs.
   - **A ≈ B ≈ -1** ⟹ NO es compile; mi diagnóstico está mal, hay que seguir buscando.
   - **C ≈ -0.65 y el assert NO salta** ⟹ el estado siempre fue un one-hot válido; solo el `==` fallaba y el fix argmax lo resuelve.
   - **C: el assert SALTA** (`reward input is not (close to) one-hot`) ⟹ bajo compile `imag_stoch` llega corrupto (aliasing de cudagraphs), no es solo el `==`; el fix argmax por sí solo no bastaría. Reintentar C con `model.compile=False` para separar.
   - Nota: con `compile=True`, el assert de `VALIDATE_ONEHOT=1` fuerza un sync y puede romper/lentificar cudagraphs. Si C crashea por eso, correrlo con `VALIDATE_ONEHOT=0` (pierde el assert pero prueba el reward) y/o `model.compile=False`. Para la corrida de producción final: `VALIDATE_ONEHOT=0`.