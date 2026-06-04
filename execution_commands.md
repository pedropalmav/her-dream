1. Correr modelo
```bash
ts -G 1 bash random_goal.sh logdir=./logdir/full_goal_dreamer_with_text/01 seed=1 mission_text=True trainer.steps=100000 trainer.update_log_every=1000
```

1.b Correr modelo en la segunda GPU (GPU 1) — usa `goal_sample=text` para muestrear goals desde el text encoder vivo:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash random_goal.sh logdir=./logdir/wm_only_random_mission/01 env=fixed_goal seed=1 mission_text=True env.goal_sample=random trainer.steps=500000 buffer=normal wm_only=True
```

2. Traerse el tensorboard
```bash 
scp -r iamonardes@barto.ing.uc.cl:/home/iamonardes/her-dream/logdir/post_train_her_from_wm_only_sample_buffer/01 ./logdir/post_train_her_from_wm_only_sample_buffer/
```

3. Correr el tensorboard
```bash
tensorboard --logdir ./logdir/post_train_her_from_wm_only_sample_buffer/01
```

4. Correr evaluación con el text_encoder al azar:
```bash
python3 eval_text_goal.py --logdir logdir/goal_dreamer_with_text/04 --episodes 10 --device cpu
```


4.b Post-training: cargar el WM congelado desde `wm_only_random_mission/01` y entrenar solo actor/critic (puedes cambiar logdir, goal_sample, buffer, steps, etc.):
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash post_train.sh \
    load_from=./logdir/wm_only_random_mission/01 \
    logdir=./logdir/post_train_from_wm_only_sample_buffer/01 \
    freeze_wm=True wm_only=False \
    env=fixed_goal mission_text=True env.goal_sample=buffer \
    buffer=normal seed=1 trainer.steps=500000 trainer.update_log_every=1000
```

5. Correr codigo de pedro:
```bash
ts -G 1 bash random_goal.sh logdir=./logdir/observed-z-goals/01 seed=1 trainer.steps=1000000 trainer.update_log_every=1000 env.goal_sample=buffer buffer=her env=fixed_goal
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
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash post_train.sh \
      load_from=./logdir/wm_only_random_mission/01 \
      logdir=./logdir/post_train_from_wm_only/01 \
      freeze_wm=True wm_only=False \
      env=fixed_goal mission_text=True env.goal_sample=random \
      buffer=normal seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

11. Post train con goals sampleados del replay buffer (misiones pasadas), sin HER:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash post_train.sh \
      load_from=./logdir/wm_only_random_mission/01 \
      logdir=./logdir/post_train_from_wm_only/02_normal_buffer_goals \
      freeze_wm=True wm_only=False \
      env=fixed_goal mission_text=True env.goal_sample=buffer \
      buffer=normal seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

12. Post train con goals sampleados del replay buffer (misiones pasadas), con HER:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash post_train.sh \
      load_from=./logdir/wm_only_random_mission/01 \
      logdir=./logdir/post_train_from_wm_only/03_her_buffer_goals \
      freeze_wm=True wm_only=False \
      env=fixed_goal mission_text=True env.goal_sample=buffer \
      buffer=her seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

13. Fase B — Destilar el text encoder sobre el WM congelado. Carga el checkpoint de `wm_only_random_mission/01`, mantiene WM y actor/critic fijos y entrena SOLO el `TextEncoderGRU` contra el posterior congelado (target estacionario). Usa el mismo `env`/`buffer` que la corrida wm_only (item 1.b) para que la distribución de datos coincida; `goal_sample=random` evita usar el text encoder (aún sin entrenar) para muestrear goals. `text_kl=1.0` porque es la única loss:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/distill_text.sh \
      load_from=./logdir/wm_only_random_mission/01 \
      logdir=./logdir/distill_text_from_wm_only/01 \
      mission_text=True \
      env=fixed_goal env.goal_sample=random buffer=normal \
      seed=1 trainer.steps=200000 trainer.update_log_every=1000 \
      model.loss_scales.text_kl=1.0
```

14. Fase C — Entrenar la política sobre el checkpoint destilado (item 13), con WM y text encoder ya entrenados y AMBOS congelados. `goal_sample=text` muestrea los goals desde el text encoder entrenado en la fase B:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/post_train.sh \
      load_from=./logdir/distill_text_from_wm_only/01 \
      logdir=./logdir/post_train_from_distill/01 \
      freeze_wm=True wm_only=False mission_text=True \
      env=fixed_goal env.goal_sample=text \
      buffer=her seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

15. Post-train del checkpoint destilado (item 13) con reward **argmax_full** (doble argmax: moda del estado imaginado vs moda del goal) y HER. Goals muestreados desde el **text encoder** entrenado (`goal_sample=text`); con `goal_type=argmax_full` el goal de texto se toma como `argmax(text_logits)`:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/post_train.sh \
      load_from=./logdir/distill_text_from_wm_only/01 \
      logdir=./logdir/post_train_from_distill/02_argmax_text_her \
      freeze_wm=True wm_only=False mission_text=True \
      env=fixed_goal env.goal_sample=text \
      buffer=her goal_type=argmax_full seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```

16. Igual que el item 15 pero con goals muestreados desde el **replay buffer** (`goal_sample=buffer`); con `goal_type=argmax_full` el goal se toma como `argmax(data["logit"])`. Reward argmax_full + HER:
```bash
  CUDA_VISIBLE_DEVICES=1 ts -G 1 bash scripts/post_train.sh \
      load_from=./logdir/distill_text_from_wm_only/01 \
      logdir=./logdir/post_train_from_distill/03_argmax_buffer_her \
      freeze_wm=True wm_only=False mission_text=True \
      env=fixed_goal env.goal_sample=buffer \
      buffer=her goal_type=argmax_full seed=1 \
      trainer.steps=500000 trainer.update_log_every=1000
```