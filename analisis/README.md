# analisis/

Scripts para analizar los runs guardados en `../logdir/<run>/metrics.jsonl`.
Usar siempre `uv run` (matplotlib vive en el venv del proyecto, no en el python del sistema).

## `plot_run_analysis.py` — figura de 4 paneles de un run

```bash
uv run python plot_run_analysis.py ../logdir/z_without_history_wm_crafter/01
# -o salida.png   ruta de salida (def. analisis_<run>.png dentro del rundir)
# --title "..."   titulo de la figura
# --window 100    ventana de la media movil (episodios)
```

Paneles: recompensa train, recompensa eval (greedy), supervivencia (largo de
episodio), y logros de la eval final (verde=alcanzado, rojo=nunca, escala log).

## `compare_runs.py` — overlay de 2+ runs

```bash
# ejecutar desde esta carpeta (importa plot_run_analysis)
uv run python compare_runs.py \
  ../logdir/original_wm_crafter/02 \
  ../logdir/z_without_history_wm_crafter/01 \
  --labels "original (con historia)" "sin historia (z)" \
  -o comparacion_original_vs_zsinhistoria.png
```

Superpone recompensa train, recompensa eval, supervivencia (todas con media
movil) y compara los logros de la eval final con barras agrupadas por run.
