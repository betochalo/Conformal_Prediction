# Resultados generados

Cada ejecución de `run_pipeline` crea aquí un subdirectorio con su configuración,
hash de datos, versiones, índices de partición, auditoría, conteos, umbrales,
probabilidades, métricas y figuras. Los resultados se versionan para que el informe
sea trazable; una ejecución nueva nunca sobrescribe una existente salvo con
`--overwrite`.

`run_2026-09-11_seed42/` es la ejecución citada en [RESULTADOS.md](../RESULTADOS.md),
en el notebook y en la presentación.

| Archivo | Contenido |
|---|---|
| `run_manifest.json` | Configuración, SHA-256 del CSV, versiones, GPU, tiempos |
| `audit.csv`, `exclusions.csv` | Auditoría de indicadores y filas excluidas |
| `split_indices.csv`, `class_counts.csv` | Bloque de cada UID y conteos por clase |
| `calibration_feasibility.csv` | Rango y viabilidad del umbral por clase y alpha |
| `thresholds.csv` | Umbrales calibrados por modelo, variante, alpha y clase |
| `metrics.csv`, `metrics_by_class.csv` | Cobertura, tamaño, abstención y errores automáticos |
| `automatic_errors.csv` | Errores entre decisiones automáticas por clase verdadera |
| `models/<modelo>/probabilities_*.csv` | Probabilidades de calibración y prueba |
| `models/mlp/training_trace.csv`, `train_report.csv` | Traza por época y aciertos antes y después |
| `models/mlp/benchmark_*.csv` | CPU frente a GPU: tiempos, parámetros, transferencias |
| `figures/` | Figuras del pipeline y del notebook |
