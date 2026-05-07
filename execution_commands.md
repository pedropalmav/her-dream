1. Correr modelo
```bash
ts -G 1 bash random_goal.sh logdir=./logdir/full_goal_dreamer_with_text/01 seed=1 mission_text=True trainer.steps=100000 trainer.update_log_every=1000
```

2. Traerse el tensorboard
```bash
scp -r iamonardes@barto.ing.uc.cl:/home/iamonardes/her-dream/logdir/goal_dreamer_with_text/04 ./logdir/goal_dreamer_with_text/
```

3. Correr el tensorboard
```bash
tensorboard --logdir ./logdir/goal_dreamer_with_text/04
tensorboard --logdir ./logdir/02
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