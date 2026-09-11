"""Construye y ejecuta notebooks/analisis_resultados.ipynb. Desde la raíz del repositorio:

    uv run --with nbformat,nbclient,ipykernel python notebooks/construir_notebook.py
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path("notebooks/analisis_resultados.ipynb")
cells = []
md = lambda s: cells.append(new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(new_code_cell(s))  # noqa: E731

md("""# Análisis de resultados: predicción conforme con abstención en AI4I 2020

Este notebook lee la ejecución guardada en `artifacts/run_2026-09-11_seed42`, muestra sus
tablas y figuras, recorre a mano el cálculo del cuantil conforme y verifica que las
métricas del informe se reconstruyen desde las probabilidades exportadas. No entrena
nada: la ejecución se reproduce con

```bash
uv sync --extra torch
uv run python -m conformal_fault_inference_with_abstention --output artifacts/run_2026-09-11_seed42 --overwrite
```

Las figuras nuevas que genera este notebook se guardan en `artifacts/<ejecución>/figures/`.
El análisis escrito está en [RESULTADOS.md](../RESULTADOS.md).""")

code("""import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

pd.set_option("display.width", 160)
pd.set_option("display.precision", 4)

ROOT = Path.cwd() if (Path.cwd() / "artifacts").exists() else Path.cwd().parent
RUN = ROOT / "artifacts" / "run_2026-09-11_seed42"
FIG = RUN / "figures"
assert RUN.exists(), f"No existe {RUN}; ejecuta primero el pipeline."
print("Ejecución:", RUN.relative_to(ROOT))""")

md("## 1. Entorno y configuración de la ejecución")
code("""manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
print("SHA-256 del CSV:", manifest["data_sha256"])
print("Filas originales / etiquetadas / excluidas:", manifest["n_raw_rows"], manifest["n_labeled_rows"], manifest["n_excluded_rows"])
pd.Series(manifest["versions"]).to_frame("versión")""")
code("""pd.Series(manifest["config"]).to_frame("valor")""")

md("## 2. Auditoría, conteos y viabilidad de calibración")
code("""pd.read_csv(RUN / "audit.csv").set_index("metric")""")
code("""pd.read_csv(RUN / "class_counts.csv", index_col=0)""")
code("""feasibility = pd.read_csv(RUN / "calibration_feasibility.csv", index_col=0)
feasibility.pivot_table(index=feasibility.index, columns="alpha", values=["rank", "finite_threshold_possible"], aggfunc="first")""")

md("""## 3. El cuantil conforme a mano

La puntuación de no conformidad es `s(x, y) = 1 − p(y | x)`. Con `n` puntuaciones de
calibración, el umbral es la puntuación de rango `⌈(n + 1)(1 − α)⌉`, sin interpolación.
Si ese rango supera `n`, el umbral es infinito y la clase entra en todos los conjuntos.
El ejemplo usa probabilidades inventadas para ver cada paso.""")
code("""from conformal_fault_inference_with_abstention.conformal import MondrianConformal, SplitConformal, conformal_quantile, conformal_rank
from conformal_fault_inference_with_abstention.decision import route_predictions
from conformal_fault_inference_with_abstention.evaluation import evaluate_sets
from conformal_fault_inference_with_abstention.scores import true_label_scores

classes = np.array(["Normal", "TWF"])
proba_cal = np.array([[0.875, 0.125], [0.75, 0.25], [0.5, 0.5], [0.9, 0.1], [0.3, 0.7]])
y_cal = np.array(["Normal", "Normal", "TWF", "Normal", "TWF"])

scores = true_label_scores(proba_cal, y_cal, classes=classes)
alpha = 0.4
n = len(scores)
print("puntuaciones de la etiqueta verdadera:", scores)
print("ordenadas:", np.sort(scores))
print(f"n = {n}, rango = ceil(({n} + 1) * (1 - {alpha})) = {conformal_rank(n, alpha=alpha)}")
print("umbral:", conformal_quantile(scores, alpha=alpha))""")
code("""split = SplitConformal(alpha=alpha).calibrate(proba_cal, y_cal, classes=classes)
proba_test = np.array([[0.8, 0.2], [0.55, 0.45], [0.05, 0.95], [0.5, 0.5]])
sets = split.predict_set(proba_test, classes=classes)
pd.DataFrame({
    "p_Normal": proba_test[:, 0], "p_TWF": proba_test[:, 1],
    "en_conjunto": [", ".join(classes[row]) or "∅" for row in sets.mask],
    "tamaño": sets.mask.sum(axis=1), "vía": route_predictions(sets),
})""")
code("""mondrian = MondrianConformal(alpha=alpha).calibrate(proba_cal, y_cal, classes=classes)
print("umbrales Mondrian por clase:", {str(c): float(t) for c, t in zip(classes, mondrian.thresholds_)})
print("  Normal: n = 3, rango", conformal_rank(3, alpha=alpha), "→ puntuación máxima de Normal")
print("  TWF:    n = 2, rango", conformal_rank(2, alpha=alpha), "> 2 → infinito")
report = evaluate_sets(np.array(["Normal", "TWF", "TWF", "Normal"]), mondrian.predict_set(proba_test, classes=classes))
report.by_class""")

md("## 4. Métricas globales en prueba")
code("""metrics = pd.read_csv(RUN / "metrics.csv")
metrics.set_index(["model", "variant", "alpha"])[["marginal_coverage", "mean_set_size", "abstention_rate", "assisted_rate", "human_rate", "automatic_count", "automatic_errors", "automatic_error_rate"]]""")
code("""for model in metrics["model"].unique():
    display(Image(filename=str(FIG / f"global_metrics_{model}.png"), width=1100))""")

md("""## 5. Cobertura por clase

La cobertura marginal puede cumplirse ignorando las clases minoritarias. Mondrian calibra
un umbral por clase y por eso las cubre; el precio es el tamaño del conjunto.""")
code("""by_class = pd.read_csv(RUN / "metrics_by_class.csv")
by_class.pivot_table(index=["model", "variant", "alpha"], columns="class", values="coverage")""")
code("""by_class.pivot_table(index=["model", "variant", "alpha"], columns="class", values="mean_set_size")""")
code("""for model in metrics["model"].unique():
    display(Image(filename=str(FIG / f"coverage_by_class_{model}.png"), width=1100))""")

md("## 6. Umbrales calibrados")
code("""thresholds = pd.read_csv(RUN / "thresholds.csv")
thresholds.pivot_table(index=["model", "variant", "alpha"], columns="class", values="threshold").style.format("{:.3e}")""")

md("""## 7. Errores entre las decisiones automáticas

La cobertura no garantiza la exactitud condicionada a decidir de forma automática. Se mide
aparte: fracción de conjuntos de tamaño 1 cuya única clase no es la verdadera.""")
code("""errors = pd.read_csv(RUN / "automatic_errors.csv")
errors[errors["alpha"] == 0.10].pivot_table(index=["model", "variant"], columns="class", values=["automatic", "errors"], aggfunc="sum")""")
code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, model in zip(axes, metrics["model"].unique()):
    sub = metrics[metrics["model"] == model].sort_values("alpha")
    x = np.arange(sub["alpha"].nunique())
    for offset, (variant, color) in enumerate((("split", BLUE), ("mondrian", ORANGE))):
        rows = sub[sub["variant"] == variant]
        values = rows["automatic_error_rate"].fillna(0).to_numpy()
        bars = ax.bar(x + (offset - 0.5) * 0.38, values, 0.36, color=color, label=variant)
        for bar, value, count in zip(bars, values, rows["automatic_count"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.1%}\\nn={count}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [f"α = {a}" for a in sorted(sub["alpha"].unique())])
    ax.set_title(f"modelo {model}", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Error entre decisiones automáticas", color=INK)
axes[0].set_ylim(0, 1.15)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=2)
fig.suptitle("Error condicionado a decidir automáticamente (prueba)", fontsize=11, color=INK, x=0.3)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = FIG / "automatic_error_rate.png"
fig.savefig(out, dpi=160)
plt.close(fig)
display(Image(filename=str(out), width=1000))
print("guardada en", out.relative_to(ROOT))""")

md("""## 8. Segundo clasificador: MLP en GPU

Traza por época al estilo del hackatón 3 y comparación CPU frente a GPU al estilo del
ejercicio de la semana 3. Los tiempos son mediciones de la sesión que generó la ejecución.""")
code("""mlp_dir = RUN / "models" / "mlp"
if mlp_dir.exists():
    trace = pd.read_csv(mlp_dir / "training_trace.csv")
    display(trace.iloc[[0, 1, 2, 49, 99, 199]])
    display(Image(filename=str(FIG / "training_trace_mlp.png"), width=1100))
    display(pd.read_csv(mlp_dir / "train_report.csv"))
    print("línea base mayoritaria en entrenamiento:", round(manifest["mlp"]["majority_baseline_train"], 4))
else:
    print("Esta ejecución no incluye el MLP.")""")
code("""if (mlp_dir / "benchmark_devices.csv").exists():
    display(pd.read_csv(mlp_dir / "benchmark_devices.csv").drop(columns="muestras_ms"))
    display(pd.read_csv(mlp_dir / "benchmark_parameters.csv"))
    display(pd.read_csv(mlp_dir / "benchmark_transfer.csv").drop(columns="muestras_ms"))
    pd.Series(manifest["mlp"]["entorno"]).to_frame("entorno")""")

md("""## 9. Verificación: reconstruir las métricas desde las probabilidades guardadas

Con `probabilities_test.csv` y `thresholds.csv` se reconstruyen los conjuntos y se comparan
las métricas con `metrics.csv`. Si coinciden, el informe es trazable hasta las
probabilidades del modelo.""")
code("""from conformal_fault_inference_with_abstention.contracts import PredictionSets

rows = []
for model in metrics["model"].unique():
    pt = pd.read_csv(RUN / "models" / model / "probabilities_test.csv", index_col=0)
    cls = np.array([c[2:] for c in pt.columns if c.startswith("p_")])
    P = pt[[f"p_{c}" for c in cls]].to_numpy()
    y = pt["y_true"].to_numpy(dtype=str)
    for (variant, a), th in thresholds[thresholds["model"] == model].groupby(["variant", "alpha"]):
        t = th.set_index("class").loc[cls, "threshold"].to_numpy()
        report = evaluate_sets(y, PredictionSets(cls, (1 - P) <= t))
        rows.append({"model": model, "variant": variant, "alpha": a,
                     "cov_reconstruida": report.marginal_coverage, "size_reconstruido": report.mean_set_size,
                     "err_auto_reconstruido": report.automatic_error_rate})
check = metrics.merge(pd.DataFrame(rows), on=["model", "variant", "alpha"])
ok = (np.allclose(check["marginal_coverage"], check["cov_reconstruida"])
      and np.allclose(check["mean_set_size"], check["size_reconstruido"])
      and np.allclose(check["automatic_error_rate"].fillna(-1), check["err_auto_reconstruido"].fillna(-1)))
print("Métricas reconstruidas coinciden con metrics.csv:", ok)
check[["model", "variant", "alpha", "marginal_coverage", "cov_reconstruida", "mean_set_size", "size_reconstruido"]]""")

md("""## 10. Lectura

- En esta partición, las coberturas marginales quedan a menos de 0.02 del nominal con ambos
  modelos. Es consistente con la garantía, que se sostiene bajo intercambiabilidad; una
  partición la ilustra, no la demuestra.
- Split conformal cumple la cobertura marginal sin cubrir TWF ni PWF. Mondrian lleva las
  cuatro fallas a cobertura 0.95 o más con α ≤ 0.10, a cambio de conjuntos más grandes.
- Entre las decisiones automáticas, Mondrian comete falsas alarmas (5.9 % hgb, 8.2 % MLP con
  α = 0.10) y split omite fallas (0.3 %, todas fallas enviadas como Normal). Cuál conviene
  depende de costos que este proyecto no fija.
- El MLP en GPU no fue más rápido que en CPU: el modelo es demasiado pequeño para que el
  cálculo supere el costo de lanzar kernels.

Detalles, referencias y limitaciones en [RESULTADOS.md](../RESULTADOS.md).""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(exist_ok=True)
client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(OUT.parent)}})
client.execute()
nbformat.write(nb, OUT)
n_err = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error")
print("guardado", OUT, "celdas:", len(nb.cells), "errores:", n_err)
