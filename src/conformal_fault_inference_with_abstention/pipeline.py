"""Punto de entrada importable para integrar las etapas del producto mínimo.

Secuencia: CSV → auditoría → etiquetas → particiones → reporte previo → modelos
entrenados una sola vez → probabilidades de calibración y prueba → split / Mondrian
→ conjuntos → métricas y figuras. Todo se exporta a output_dir.
"""

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .conformal import MondrianConformal, SplitConformal
from .contracts import CLASSES, EvaluationReport, LabelPolicy
from .data import audit_failures, build_labels, exclusion_mask, load_raw
from .evaluation import evaluate_sets
from .model import build_model
from .splitting import calibration_feasibility, class_counts, split_data

VARIANTS = ("split", "mondrian")
MODELS = ("hgb", "mlp")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"


@dataclass(frozen=True)
class RunConfig:
    """Decisiones fijadas tras la auditoría; ver data/README.md para los valores reales.

    models: 'hgb' es el clasificador base aprobado; 'mlp' es la extensión en PyTorch
    y requiere el extra opcional torch. Los parámetros mlp_* solo aplican a 'mlp'.
    mlp_benchmark_repetitions > 0 entrena además el MLP ese número de veces en cada
    dispositivo disponible (CPU y CUDA) para comparar tiempos y resultados, como en
    el ejercicio de la semana 3; 0 omite la comparación.
    overwrite: por defecto se rechaza un output_dir que ya contiene archivos, para no
    mezclar resultados de ejecuciones con configuraciones distintas.
    """

    data_path: Path
    output_dir: Path
    label_policy: LabelPolicy
    calibration_size: float
    test_size: float
    alphas: tuple[float, ...]
    random_state: int
    models: tuple[str, ...] = ("hgb",)
    mlp_hidden: tuple[int, ...] = (64, 64)
    mlp_epochs: int = 200
    mlp_batch_size: int = 256
    mlp_learning_rate: float = 1e-3
    mlp_device: str = "auto"
    mlp_benchmark_repetitions: int = 0
    overwrite: bool = False


Results = dict[str, dict[str, dict[float, EvaluationReport]]]


def _validate_config(config: RunConfig) -> None:
    if not config.alphas:
        raise ValueError("alphas no puede estar vacío.")
    if any(not 0 < a < 1 for a in config.alphas):
        raise ValueError("Cada alpha debe estar entre 0 y 1, sin incluir extremos.")
    if len(set(config.alphas)) != len(config.alphas):
        raise ValueError("alphas contiene valores repetidos.")
    if not config.models:
        raise ValueError("models no puede estar vacío.")
    unknown = set(config.models) - set(MODELS)
    if unknown:
        raise ValueError(f"Modelos desconocidos: {sorted(unknown)}; admitidos {MODELS}.")
    if len(set(config.models)) != len(config.models):
        raise ValueError("models contiene valores repetidos.")
    if config.mlp_benchmark_repetitions < 0:
        raise ValueError("mlp_benchmark_repetitions no puede ser negativo.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions(models: tuple[str, ...]) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": metadata.version("numpy"),
        "pandas": metadata.version("pandas"),
        "scikit-learn": metadata.version("scikit-learn"),
        "matplotlib": metadata.version("matplotlib"),
        "package": metadata.version("conformal-fault-inference-with-abstention"),
    }
    if "mlp" in models:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda_available"] = str(torch.cuda.is_available())
        versions["cuda_version"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
    return versions


def _build(name: str, config: RunConfig) -> Pipeline:
    if name == "hgb":
        return build_model(random_state=config.random_state)
    from .torch_model import build_torch_model

    return build_torch_model(
        random_state=config.random_state,
        hidden=config.mlp_hidden,
        epochs=config.mlp_epochs,
        batch_size=config.mlp_batch_size,
        learning_rate=config.mlp_learning_rate,
        device=config.mlp_device,
    )


def _probability_frame(index: pd.Index, proba: np.ndarray, classes: np.ndarray, y) -> pd.DataFrame:
    frame = pd.DataFrame(proba, index=index, columns=[f"p_{c}" for c in classes])
    frame.insert(0, "y_true", y)
    return frame


def _report_rows(model: str, variant: str, alpha: float, report: EvaluationReport) -> dict:
    return {
        "model": model,
        "variant": variant,
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "n_samples": report.n_samples,
        "marginal_coverage": report.marginal_coverage,
        "mean_set_size": report.mean_set_size,
        "abstention_rate": report.abstention_rate,
        "assisted_rate": report.assisted_rate,
        "human_rate": report.human_rate,
        "automatic_count": report.automatic_count,
        "automatic_errors": report.automatic_errors,
        "automatic_error_rate": report.automatic_error_rate,
    }


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def _plot_coverage_by_class(by_class: pd.DataFrame, model: str, alphas, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(CLASSES))
    width = 0.38
    for ax, alpha in zip(axes, alphas, strict=True):
        for offset, (variant, color) in enumerate(zip(VARIANTS, (BLUE, ORANGE), strict=True)):
            rows = by_class[(by_class["variant"] == variant) & (by_class["alpha"] == alpha)]
            values = rows.set_index("class").reindex(CLASSES)["coverage"].to_numpy()
            bars = ax.bar(
                x + (offset - 0.5) * width, values, width * 0.94, color=color, label=variant
            )
            for bar, value in zip(bars, values, strict=True):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.01,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=INK,
                    )
        ax.axhline(1 - alpha, color=INK, linewidth=1, linestyle="--")
        ax.text(
            len(CLASSES) - 0.5, 1 - alpha + 0.01, f"1-α = {1 - alpha:.2f}", ha="right", fontsize=8
        )
        ax.set_xticks(x, CLASSES)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"α = {alpha}", fontsize=10, color=INK)
        _style(ax)
    axes[0].set_ylabel("Cobertura por clase en prueba", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=2)
    fig.suptitle(f"Cobertura por clase · modelo {model}", fontsize=11, color=INK, x=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_size_and_abstention(metrics: pd.DataFrame, model: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    columns = ("marginal_coverage", "mean_set_size", "abstention_rate")
    titles = ("Cobertura marginal", "Tamaño medio del conjunto", "Tasa de abstención")
    for ax, column, title in zip(axes, columns, titles, strict=True):
        for variant, color in zip(VARIANTS, (BLUE, ORANGE), strict=True):
            rows = metrics[metrics["variant"] == variant].sort_values("alpha")
            ax.plot(
                rows["alpha"], rows[column], marker="o", color=color, linewidth=2, label=variant
            )
            for alpha, value in zip(rows["alpha"], rows[column], strict=True):
                ax.annotate(
                    f"{value:.2f}",
                    (alpha, value),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    color=INK,
                )
        if column == "marginal_coverage":
            alphas = sorted(metrics["alpha"].unique())
            ax.plot(
                alphas, [1 - a for a in alphas], color=INK, linestyle="--", linewidth=1, label="1-α"
            )
        ax.set_xlabel("α", color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=3)
    fig.suptitle(f"Métricas globales en prueba · modelo {model}", fontsize=11, color=INK, x=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_training_trace(history: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    series = (
        ("loss_antes", "Pérdida media antes del paso"),
        ("grad_norm", "Norma media del gradiente"),
        ("change_norm", "Norma del cambio de parámetros"),
    )
    for ax, (column, title) in zip(axes, series, strict=True):
        ax.plot(history["paso"], history[column], color=BLUE, linewidth=2)
        ax.set_xlabel("Época", color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        _style(ax)
    fig.suptitle("Traza de entrenamiento del MLP", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_pipeline(config: RunConfig) -> Results:
    """Auditar, etiquetar, partir, reportar conteos, entrenar, calibrar y evaluar.

    Guarda auditoría y conteos antes del fit. Entrena cada modelo una sola vez y
    reutiliza probabilidades y particiones para ambas variantes y todos los alpha.
    Retorna {modelo: {variante: {alpha: reporte}}}. Nada se selecciona con prueba.
    """
    _validate_config(config)
    data_path = Path(config.data_path)
    out = Path(config.output_dir)
    if out.exists() and any(out.iterdir()) and not config.overwrite:
        raise FileExistsError(
            f"{out} ya contiene archivos. Usa otro output_dir o overwrite=True para "
            "reemplazar sus resultados."
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    raw = load_raw(data_path)
    audit = audit_failures(raw)
    audit.to_csv(out / "audit.csv", index=False)
    exclusions = exclusion_mask(raw, policy=config.label_policy)
    exclusions[exclusions.any(axis=1)].to_csv(out / "exclusions.csv")
    data = build_labels(raw, policy=config.label_policy)

    splits = split_data(
        data,
        calibration_size=config.calibration_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    indices = pd.concat(
        [
            pd.Series(name, index=block.X.index, name="block")
            for name, block in (
                ("train", splits.train),
                ("calibration", splits.calibration),
                ("test", splits.test),
            )
        ]
    ).sort_index()
    indices.to_csv(out / "split_indices.csv")
    counts = class_counts(splits)
    counts.to_csv(out / "class_counts.csv")
    feasibility = pd.concat(
        [calibration_feasibility(splits, alpha=alpha) for alpha in config.alphas]
    )
    feasibility.to_csv(out / "calibration_feasibility.csv")

    config_dict = asdict(config)
    config_dict["data_path"] = str(data_path)
    config_dict["output_dir"] = str(out)
    manifest = {
        "started_utc": datetime.now(UTC).isoformat(),
        "data_sha256": _sha256(data_path),
        "n_raw_rows": int(len(raw)),
        "n_labeled_rows": int(len(data.y)),
        "n_excluded_rows": int(exclusions.any(axis=1).sum()),
        "versions": _versions(config.models),
        "config": config_dict,
    }

    results: Results = {}
    metric_rows: list[dict] = []
    class_rows: list[dict] = []
    error_rows: list[pd.DataFrame] = []
    threshold_rows: list[dict] = []
    for name in config.models:
        model_dir = out / "models" / name
        model_dir.mkdir(parents=True, exist_ok=True)
        model = _build(name, config)
        model.fit(splits.train.X, splits.train.y)
        classes = np.asarray(model.classes_, dtype=np.str_)
        proba_cal = model.predict_proba(splits.calibration.X)
        proba_test = model.predict_proba(splits.test.X)
        _probability_frame(
            splits.calibration.X.index, proba_cal, classes, splits.calibration.y
        ).to_csv(model_dir / "probabilities_calibration.csv")
        _probability_frame(splits.test.X.index, proba_test, classes, splits.test.y).to_csv(
            model_dir / "probabilities_test.csv"
        )
        if name == "mlp":
            classifier = model.named_steps["classifier"]
            classifier.history_.to_csv(model_dir / "training_trace.csv", index=False)
            classifier.train_report_.to_csv(model_dir / "train_report.csv", index=False)
            _plot_training_trace(classifier.history_, out / "figures" / "training_trace_mlp.png")
            from .torch_model import benchmark_devices, benchmark_transfer, environment_record

            manifest["mlp"] = {
                "device": str(classifier.device_),
                "fit_seconds": classifier.fit_seconds_,
                "n_parameters": classifier.n_parameters_,
                "majority_baseline_train": classifier.majority_baseline_,
                "initial_loss": float(classifier.history_["loss_antes"].iloc[0]),
                "final_loss": float(classifier.history_["loss_antes"].iloc[-1]),
                "entorno": environment_record(),
            }
            if config.mlp_benchmark_repetitions > 0:
                preprocessing = model.named_steps["preprocessing"]
                X_train = preprocessing.transform(splits.train.X)
                timings, comparison = benchmark_devices(
                    X_train,
                    splits.train.y,
                    repetitions=config.mlp_benchmark_repetitions,
                    hidden=config.mlp_hidden,
                    epochs=config.mlp_epochs,
                    batch_size=config.mlp_batch_size,
                    learning_rate=config.mlp_learning_rate,
                    random_state=config.random_state,
                )
                timings.to_csv(model_dir / "benchmark_devices.csv", index=False)
                comparison.to_csv(model_dir / "benchmark_parameters.csv", index=False)
                transfer = benchmark_transfer(classifier, preprocessing.transform(splits.test.X))
                transfer.to_csv(model_dir / "benchmark_transfer.csv", index=False)
                manifest["mlp"]["benchmark"] = {
                    "repeticiones": config.mlp_benchmark_repetitions,
                    "dispositivos": timings["device"].tolist(),
                    "mediana_ms": dict(
                        zip(timings["device"], timings["mediana_ms"].round(1), strict=True)
                    ),
                    "parametros_cercanos": dict(
                        zip(comparison["device"], comparison["parametros_cercanos"], strict=True)
                    ),
                }

        results[name] = {variant: {} for variant in VARIANTS}
        for alpha in config.alphas:
            predictors = {
                "split": SplitConformal(alpha=alpha),
                "mondrian": MondrianConformal(alpha=alpha),
            }
            for variant, predictor in predictors.items():
                predictor.calibrate(proba_cal, splits.calibration.y, classes=classes)
                sets = predictor.predict_set(proba_test, classes=classes)
                report = evaluate_sets(splits.test.y, sets)
                results[name][variant][alpha] = report
                metric_rows.append(_report_rows(name, variant, alpha, report))
                by_class = report.by_class.copy()
                by_class.insert(0, "alpha", alpha)
                by_class.insert(0, "variant", variant)
                by_class.insert(0, "model", name)
                class_rows.append(by_class)
                errors = report.automatic_errors_by_class.copy()
                errors.insert(0, "alpha", alpha)
                errors.insert(0, "variant", variant)
                errors.insert(0, "model", name)
                error_rows.append(errors)
                thresholds = (
                    [predictor.threshold_] * len(classes)
                    if variant == "split"
                    else list(predictor.thresholds_)
                )
                threshold_rows += [
                    {"model": name, "variant": variant, "alpha": alpha, "class": c, "threshold": t}
                    for c, t in zip(classes, thresholds, strict=True)
                ]

        model_metrics = pd.DataFrame([r for r in metric_rows if r["model"] == name])
        model_by_class = pd.concat([c for c in class_rows if c["model"].iloc[0] == name])
        _plot_coverage_by_class(
            model_by_class, name, config.alphas, out / "figures" / f"coverage_by_class_{name}.png"
        )
        _plot_size_and_abstention(
            model_metrics, name, out / "figures" / f"global_metrics_{name}.png"
        )

    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    pd.concat(class_rows).to_csv(out / "metrics_by_class.csv", index=False)
    pd.concat(error_rows).to_csv(out / "automatic_errors.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out / "thresholds.csv", index=False)
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    manifest["argv"] = sys.argv
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results
