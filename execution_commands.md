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
scp -r iamonardes@barto.ing.uc.cl:/home/iamonardes/her-dream/logdir/wm_only_random_mission/01 ./logdir/wm_only_random_mission/
```

3. Correr el tensorboard
```bash
tensorboard --logdir ./logdir/wm_only_random_mission/01
```

4. Correr evaluación con el text_encoder al azar:
```bash
python3 eval_text_goal.py --logdir logdir/goal_dreamer_with_text/04 --episodes 10 --device cpu
```


4.b Post-training: cargar el WM congelado desde `wm_only_random_mission/01` y entrenar solo actor/critic (puedes cambiar logdir, goal_sample, buffer, steps, etc.):
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 python3 post_train.py \
    load_from=./logdir/wm_only_random_mission/01 \
    logdir=./logdir/post_train_from_wm_only/01 \
    freeze_wm=True wm_only=False \
    env=fixed_goal mission_text=True env.goal_sample=text \
    buffer=her seed=1 trainer.steps=500000 trainer.update_log_every=1000
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