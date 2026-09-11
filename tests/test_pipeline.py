"""Prueba de integración con un CSV sintético pequeño, sin descargas."""

import json

import numpy as np
import pandas as pd
import pytest

from conformal_fault_inference_with_abstention.contracts import CLASSES, LabelPolicy
from conformal_fault_inference_with_abstention.pipeline import RunConfig, run_pipeline

POLICY = LabelPolicy(priority=("TWF", "PWF", "OSF", "HDF"), rnf_policy="exclude_only_rnf")


def synthetic_csv(path, n: int = 900, seed: int = 0) -> None:
    """Generar filas con modos derivados de umbrales, parecidos a AI4I."""
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    process = air + rng.normal(10, 1, n)
    speed = rng.normal(1540, 180, n).clip(1200, 2900)
    torque = rng.normal(40, 10, n).clip(5, 80)
    wear = rng.integers(0, 250, n)
    power = torque * speed * 2 * np.pi / 60
    twf = (wear > 235) & (rng.random(n) < 0.6)
    hdf = ((process - air) < 8.6) & (speed < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf = (wear * torque) > 11000
    rnf = rng.random(n) < 0.002
    frame = pd.DataFrame(
        {
            "UID": np.arange(1, n + 1),
            "Product ID": [f"L{i:05d}" for i in range(n)],
            "Type": rng.choice(["L", "M", "H"], n, p=[0.5, 0.3, 0.2]),
            "Air temperature": air.round(1),
            "Process temperature": process.round(1),
            "Rotational speed": speed.astype(int),
            "Torque": torque.round(1),
            "Tool wear": wear,
            "TWF": twf.astype(int),
            "HDF": hdf.astype(int),
            "PWF": pwf.astype(int),
            "OSF": osf.astype(int),
            "RNF": rnf.astype(int),
        }
    )
    frame.insert(8, "Machine failure", frame[["TWF", "HDF", "PWF", "OSF"]].max(axis=1))
    # Forzar algunos casos por clase y una falla sin modo para ejercitar las exclusiones.
    for column in ("TWF", "HDF", "PWF", "OSF"):
        if frame[column].sum() < 12:
            frame.loc[rng.choice(n, 12, replace=False), column] = 1
    frame["Machine failure"] = frame[["TWF", "HDF", "PWF", "OSF"]].max(axis=1)
    frame.loc[0, ["TWF", "HDF", "PWF", "OSF", "RNF"]] = 0
    frame.loc[0, "Machine failure"] = 1
    frame.to_csv(path, index=False)


def make_config(tmp_path, **overrides) -> RunConfig:
    csv = tmp_path / "synthetic.csv"
    synthetic_csv(csv)
    base = dict(
        data_path=csv,
        output_dir=tmp_path / "run",
        label_policy=POLICY,
        calibration_size=0.3,
        test_size=0.2,
        alphas=(0.1, 0.3),
        random_state=0,
        models=("hgb",),
    )
    base.update(overrides)
    return RunConfig(**base)


def test_run_pipeline_exports_every_artifact(tmp_path):
    config = make_config(tmp_path)
    results = run_pipeline(config)
    out = config.output_dir

    assert set(results) == {"hgb"}
    assert set(results["hgb"]) == {"split", "mondrian"}
    for variant in ("split", "mondrian"):
        assert set(results["hgb"][variant]) == {0.1, 0.3}
        report = results["hgb"][variant][0.1]
        assert 0 <= report.marginal_coverage <= 1
        assert set(report.by_class["class"]) == set(CLASSES)

    expected = [
        "audit.csv",
        "exclusions.csv",
        "split_indices.csv",
        "class_counts.csv",
        "calibration_feasibility.csv",
        "metrics.csv",
        "metrics_by_class.csv",
        "automatic_errors.csv",
        "thresholds.csv",
        "run_manifest.json",
        "models/hgb/probabilities_calibration.csv",
        "models/hgb/probabilities_test.csv",
        "figures/coverage_by_class_hgb.png",
        "figures/global_metrics_hgb.png",
    ]
    for name in expected:
        assert (out / name).exists(), name

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["data_sha256"]) == 64
    assert manifest["n_raw_rows"] == 900
    assert manifest["n_raw_rows"] == manifest["n_labeled_rows"] + manifest["n_excluded_rows"]
    assert manifest["config"]["alphas"] == [0.1, 0.3]
    assert "scikit-learn" in manifest["versions"]

    indices = pd.read_csv(out / "split_indices.csv", index_col=0)
    assert indices.index.is_unique
    assert len(indices) == manifest["n_labeled_rows"]
    assert set(indices["block"]) == {"train", "calibration", "test"}

    metrics = pd.read_csv(out / "metrics.csv")
    assert len(metrics) == 4
    assert (metrics["automatic_errors"] <= metrics["automatic_count"]).all()
    errors = pd.read_csv(out / "automatic_errors.csv")
    totals = errors.groupby(["model", "variant", "alpha"])["automatic"].sum()
    for row in metrics.itertuples():
        assert totals.loc[(row.model, row.variant, row.alpha)] == row.automatic_count
    assert set(metrics["variant"]) == {"split", "mondrian"}
    np.testing.assert_allclose(
        metrics["abstention_rate"], metrics["assisted_rate"] + metrics["human_rate"], atol=1e-12
    )

    thresholds = pd.read_csv(out / "thresholds.csv")
    split = thresholds[(thresholds["variant"] == "split") & (thresholds["alpha"] == 0.1)]
    assert split["threshold"].nunique() == 1
    assert len(thresholds) == 2 * 2 * len(CLASSES)

    probabilities = pd.read_csv(out / "models/hgb/probabilities_test.csv", index_col=0)
    assert set(probabilities.index) == set(indices.index[indices["block"] == "test"])
    np.testing.assert_allclose(probabilities.drop(columns="y_true").sum(axis=1), 1, atol=1e-6)


def test_run_pipeline_is_reproducible(tmp_path):
    a = run_pipeline(make_config(tmp_path, output_dir=tmp_path / "a"))
    b = run_pipeline(make_config(tmp_path, output_dir=tmp_path / "b"))
    for variant in ("split", "mondrian"):
        for alpha in (0.1, 0.3):
            assert a["hgb"][variant][alpha].marginal_coverage == pytest.approx(
                b["hgb"][variant][alpha].marginal_coverage
            )
    metrics_a = pd.read_csv(tmp_path / "a" / "metrics.csv")
    metrics_b = pd.read_csv(tmp_path / "b" / "metrics.csv")
    pd.testing.assert_frame_equal(metrics_a, metrics_b)


@pytest.mark.parametrize(
    "overrides",
    [
        {"alphas": ()},
        {"alphas": (0.1, 1.0)},
        {"alphas": (0.1, 0.1)},
        {"models": ()},
        {"models": ("hgb", "otro")},
        {"models": ("hgb", "hgb")},
    ],
)
def test_run_pipeline_rejects_invalid_config(tmp_path, overrides):
    with pytest.raises(ValueError):
        run_pipeline(make_config(tmp_path, **overrides))


def test_run_pipeline_with_mlp_exports_training_trace(tmp_path):
    pytest.importorskip("torch")
    config = make_config(tmp_path, models=("hgb", "mlp"), mlp_epochs=3, mlp_device="cpu")
    results = run_pipeline(config)
    out = config.output_dir
    assert set(results) == {"hgb", "mlp"}
    trace = pd.read_csv(out / "models/mlp/training_trace.csv")
    assert len(trace) == 3
    assert (out / "models/mlp/train_report.csv").exists()
    assert (out / "figures/training_trace_mlp.png").exists()
    assert (out / "figures/coverage_by_class_mlp.png").exists()
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mlp"]["device"] == "cpu"
    assert "torch" in manifest["versions"]
    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["model"]) == {"hgb", "mlp"}


def test_run_pipeline_mlp_benchmark_exports_device_comparison(tmp_path):
    pytest.importorskip("torch")
    config = make_config(
        tmp_path,
        models=("mlp",),
        mlp_epochs=2,
        mlp_device="cpu",
        mlp_benchmark_repetitions=2,
    )
    run_pipeline(config)
    out = config.output_dir
    timings = pd.read_csv(out / "models/mlp/benchmark_devices.csv")
    assert "cpu" in timings["device"].tolist()
    assert (timings["repeticiones"] == 2).all()
    assert (out / "models/mlp/benchmark_parameters.csv").exists()
    assert (out / "models/mlp/benchmark_transfer.csv").exists()
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mlp"]["fit_seconds"] > 0
    assert manifest["mlp"]["entorno"]["dtype"] == "float32"
    assert manifest["mlp"]["benchmark"]["repeticiones"] == 2
    with pytest.raises(ValueError):
        run_pipeline(make_config(tmp_path, mlp_benchmark_repetitions=-1))


def test_run_pipeline_refuses_non_empty_output_dir(tmp_path):
    config = make_config(tmp_path)
    run_pipeline(config)
    with pytest.raises(FileExistsError):
        run_pipeline(config)
    run_pipeline(make_config(tmp_path, overwrite=True))
