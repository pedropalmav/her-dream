1. Correr modelo
```bash
ts -G 1 bash random_goal.sh logdir=./logdir/full_goal_dreamer_with_text/01 seed=1 mission_text=True trainer.steps=100000 trainer.update_log_every=1000
```

1.b Correr modelo en la segunda GPU (GPU 1) — usa `goal_sample=text` para muestrear goals desde el text encoder vivo:
```bash
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash random_goal.sh logdir=./logdir/text_goal_sample_normal_buffer/01 seed=1 mission_text=True env.goal_sample=text trainer.steps=500000 buffer=normal wm_only=False
```

2. Traerse el tensorboard
```bash 
scp -r iamonardes@barto.ing.uc.cl:/home/iamonardes/her-dream/logdir/text_goal_sample/01 ./logdir/text_goal_sample
```

3. Correr el tensorboard
```bash
tensorboard --logdir ./logdir/text_goal_sample/01
```

4. Correr evaluación con el text_encoder al azar:
```bash
python3 eval_text_goal.py --logdir logdir/goal_dreamer_with_text/04 --episodes 10 --device cpu
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