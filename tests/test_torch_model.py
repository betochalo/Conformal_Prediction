"""Pruebas del MLP en PyTorch. Se omiten si el extra torch no está instalado."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from conformal_fault_inference_with_abstention.contracts import FEATURE_COLUMNS

torch = pytest.importorskip("torch")
torch_model = pytest.importorskip("conformal_fault_inference_with_abstention.torch_model")


def make_frame(n: int = 80, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=FEATURE_COLUMNS[1:])
    X.insert(0, "Type", np.tile(["L", "M"], n // 2))
    y = np.where(X["Torque"] > 0, "TWF", "Normal").astype(np.str_)
    return X, y


def test_mlp_trains_on_cpu_and_returns_valid_probabilities():
    X, y = make_frame()
    model = torch_model.build_torch_model(random_state=1, epochs=5, batch_size=16, device="cpu")
    with pytest.raises(NotFittedError):
        model.predict_proba(X)
    model.fit(X, y)
    classifier = model.named_steps["classifier"]
    assert list(classifier.classes_) == ["Normal", "TWF"]
    assert str(classifier.device_) == "cpu"
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_array_equal(model.predict(X), classifier.classes_[proba.argmax(axis=1)])


def test_mlp_history_matches_hackathon_trace():
    X, y = make_frame()
    model = torch_model.build_torch_model(random_state=1, epochs=6, batch_size=16, device="cpu")
    model.fit(X, y)
    classifier = model.named_steps["classifier"]
    history = classifier.history_
    assert list(history.columns) == list(torch_model.TRACE_COLUMNS)
    assert len(history) == 6
    assert history["paso"].tolist() == [1, 2, 3, 4, 5, 6]
    assert np.isfinite(history.to_numpy()).all()
    assert (history["grad_norm"] >= 0).all()
    assert (history["change_norm"] > 0).all()
    report = classifier.train_report_
    assert list(report.columns) == ["class", "support", "accuracy_before", "accuracy_after"]
    assert report["support"].sum() == len(X)
    assert 0 < classifier.majority_baseline_ < 1
    assert classifier.n_parameters_ > 0


def test_mlp_is_reproducible_with_seed_on_cpu():
    X, y = make_frame()
    a = torch_model.build_torch_model(random_state=3, epochs=4, device="cpu").fit(X, y)
    b = torch_model.build_torch_model(random_state=3, epochs=4, device="cpu").fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-6)


def test_mlp_scaler_is_fit_only_on_training_data():
    X, y = make_frame()
    model = torch_model.build_torch_model(random_state=1, epochs=2, device="cpu").fit(X, y)
    scaler = model.named_steps["preprocessing"].named_transformers_["numeric"]
    mean_before = scaler.mean_.copy()
    probe = X.iloc[:5].copy() * 100
    probe["Type"] = "unknown"
    proba = model.predict_proba(probe)
    np.testing.assert_array_equal(scaler.mean_, mean_before)
    assert proba.shape == (5, 2)


def test_mlp_rejects_invalid_inputs():
    X, y = make_frame()
    with pytest.raises(ValueError):
        torch_model.build_torch_model(random_state=1, epochs=0, device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="dos clases"):
        torch_model.build_torch_model(random_state=1, epochs=1, device="cpu").fit(
            X, np.array(["Normal"] * len(X))
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no disponible")
def test_mlp_auto_device_uses_cuda_when_available():
    X, y = make_frame()
    model = torch_model.build_torch_model(random_state=1, epochs=3, device="auto").fit(X, y)
    assert model.named_steps["classifier"].device_.type == "cuda"
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-12)


def test_environment_record_and_benchmark_on_cpu():
    X, y = make_frame()
    record = torch_model.environment_record()
    assert record["pytorch"] == torch.__version__
    assert record["dtype"] == "float32"
    assert "hilos_cpu" in record

    from sklearn.preprocessing import StandardScaler

    X_num = StandardScaler().fit_transform(X.drop(columns="Type"))
    timings, comparison = torch_model.benchmark_devices(
        X_num, y, repetitions=2, devices=("cpu",), epochs=3, random_state=1
    )
    assert timings["device"].tolist() == ["cpu"]
    assert timings.loc[0, "repeticiones"] == 2
    assert len(timings.loc[0, "muestras_ms"]) == 2
    assert timings.loc[0, "min_ms"] <= timings.loc[0, "mediana_ms"] <= timings.loc[0, "max_ms"]
    assert timings.loc[0, "loss_finita"]
    assert comparison.loc[0, "parametros_cercanos"]
    assert comparison.loc[0, "max_diferencia_abs_parametros"] == 0
    assert comparison.loc[0, "misma_clase_predicha"] == 1.0

    model = torch_model.TorchMLPClassifier(device="cpu", epochs=2, random_state=1).fit(X_num, y)
    assert model.fit_seconds_ > 0
    transfer = torch_model.benchmark_transfer(model, X_num, repetitions=3)
    assert transfer["modo"].tolist() == ["residente", "con_transferencias"]
    assert (transfer["n_filas"] == len(X)).all()

    with pytest.raises(ValueError):
        torch_model.benchmark_devices(X_num, y, repetitions=0, devices=("cpu",), epochs=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no disponible")
def test_benchmark_compares_cpu_and_cuda_within_tolerance():
    X, y = make_frame(n=200)
    from sklearn.preprocessing import StandardScaler

    X_num = StandardScaler().fit_transform(X.drop(columns="Type"))
    timings, comparison = torch_model.benchmark_devices(
        X_num, y, repetitions=1, devices=("cpu", "cuda"), epochs=5, random_state=1
    )
    assert timings["device"].tolist() == ["cpu", "cuda"]
    cuda = comparison.set_index("device").loc["cuda"]
    assert cuda["referencia"] == "cpu"
    assert cuda["max_diferencia_abs_probabilidades"] < 0.05
    assert cuda["misma_clase_predicha"] > 0.95
