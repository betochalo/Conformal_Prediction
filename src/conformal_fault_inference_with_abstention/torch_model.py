"""Segundo clasificador (extensión): MLP en PyTorch entrenado en GPU si hay CUDA.

Expone la misma interfaz que el modelo base (fit, predict_proba, classes_) para que
la capa conforme no distinga entre ambos. La traza de entrenamiento sigue el registro
del hackatón 3: pérdida antes del paso, norma del gradiente y norma del cambio de
parámetros, junto con una tabla de aciertos por clase antes y después de entrenar.
La comparación CPU frente a GPU sigue el ejercicio de la semana 3: mediana de varias
repeticiones, pérdida inicial y final por dispositivo, parámetros comparados con
tolerancia y registro del entorno. Requiere el extra opcional `torch`.
"""

import platform
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from .contracts import FEATURE_COLUMNS

TRACE_COLUMNS = ("paso", "loss_antes", "grad_norm", "change_norm", "train_accuracy")


def resolve_device(device: str) -> torch.device:
    """'auto' usa CUDA si está disponible; cualquier otro valor se pasa a torch."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Perceptrón multicapa con entropía cruzada y Adam, sin parada temprana.

    No usa calibración ni prueba durante el ajuste: el número de épocas es fijo.
    history_ registra por época la pérdida media antes de cada actualización, la
    norma media del gradiente, la norma del cambio total de parámetros y la
    exactitud de entrenamiento al cierre de la época.
    """

    def __init__(
        self,
        *,
        hidden: tuple[int, ...] = (64, 64),
        epochs: int = 200,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        random_state: int = 0,
        device: str = "auto",
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

    def _build_network(self, n_features: int, n_classes: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        width = n_features
        for units in self.hidden:
            layers += [nn.Linear(width, units), nn.ReLU()]
            width = units
        layers.append(nn.Linear(width, n_classes))
        return nn.Sequential(*layers)

    def _parameter_vector(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in self.network_.parameters()])

    def _class_accuracy(self, X: torch.Tensor, y: np.ndarray) -> pd.Series:
        predicted = self._logits(X).argmax(dim=1).cpu().numpy()
        frame = pd.DataFrame({"y": y, "hit": predicted == y})
        return frame.groupby("y")["hit"].mean().reindex(range(len(self.classes_)))

    def _logits(self, X: torch.Tensor) -> torch.Tensor:
        self.network_.eval()
        with torch.no_grad():
            return self.network_(X)

    def fit(self, X, y) -> "TorchMLPClassifier":
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs y batch_size deben ser enteros positivos.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X debe ser una matriz no vacía.")
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("X e y deben tener la misma cantidad de filas.")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Se necesitan al menos dos clases para entrenar.")

        torch.manual_seed(self.random_state)
        self.device_ = resolve_device(self.device)
        started = time.perf_counter()
        self.network_ = self._build_network(X.shape[1], len(self.classes_)).to(self.device_)
        self.n_parameters_ = int(sum(p.numel() for p in self.network_.parameters()))

        features = torch.from_numpy(X).to(self.device_)
        targets = torch.from_numpy(encoded.astype(np.int64)).to(self.device_)
        optimizer = torch.optim.Adam(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator(device="cpu").manual_seed(self.random_state)

        accuracy_before = self._class_accuracy(features, encoded)
        history: list[dict[str, float]] = []
        n = len(features)
        for epoch in range(1, self.epochs + 1):
            self.network_.train()
            start = self._parameter_vector()
            losses: list[float] = []
            grad_norms: list[float] = []
            permutation = torch.randperm(n, generator=generator).to(self.device_)
            for offset in range(0, n, self.batch_size):
                batch = permutation[offset : offset + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.network_(features[batch]), targets[batch])
                loss.backward()
                grad_norm = torch.sqrt(
                    sum((p.grad.detach() ** 2).sum() for p in self.network_.parameters())
                )
                optimizer.step()
                losses.append(float(loss.detach()))
                grad_norms.append(float(grad_norm))
            accuracy = float((self._logits(features).argmax(dim=1) == targets).float().mean())
            history.append(
                {
                    "paso": epoch,
                    "loss_antes": float(np.mean(losses)),
                    "grad_norm": float(np.mean(grad_norms)),
                    "change_norm": float(
                        torch.linalg.vector_norm(self._parameter_vector() - start)
                    ),
                    "train_accuracy": accuracy,
                }
            )
        if self.device_.type == "cuda":
            torch.cuda.synchronize(self.device_)
        self.fit_seconds_ = time.perf_counter() - started
        if not all(np.isfinite(list(row.values())).all() for row in history):
            raise RuntimeError("El entrenamiento produjo valores no finitos.")
        self.history_ = pd.DataFrame(history, columns=TRACE_COLUMNS)

        accuracy_after = self._class_accuracy(features, encoded)
        support = pd.Series(encoded).value_counts().reindex(range(len(self.classes_)), fill_value=0)
        self.train_report_ = pd.DataFrame(
            {
                "class": self.classes_,
                "support": support.to_numpy(),
                "accuracy_before": accuracy_before.to_numpy(),
                "accuracy_after": accuracy_after.to_numpy(),
            }
        )
        self.majority_baseline_ = float(support.max() / len(encoded))
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "network_"):
            raise NotFittedError("El MLP no está entrenado; llama a fit primero.")

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.network_[0].in_features:
            raise ValueError("X debe tener las mismas columnas que en el ajuste.")
        features = torch.from_numpy(X).to(self.device_)
        logits = self._logits(features).double()
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_torch_model(*, random_state: int, **mlp_params) -> Pipeline:
    """Preprocesamiento de Type y escalado numérico ajustados solo en entrenamiento.

    A diferencia de los árboles, el MLP necesita variables escaladas; el escalador
    forma parte del Pipeline y se ajusta únicamente al invocar fit con entrenamiento.
    """
    preprocessing = ColumnTransformer(
        [
            ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Type"]),
            ("numeric", StandardScaler(), list(FEATURE_COLUMNS[1:])),
        ],
        remainder="drop",
    )
    classifier = TorchMLPClassifier(random_state=random_state, **mlp_params)
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def environment_record() -> dict[str, object]:
    """Registro del entorno como en el ejercicio de la semana 3."""
    record: dict[str, object] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "plataforma": platform.platform(),
        "cpu": platform.processor(),
        "hilos_cpu": torch.get_num_threads(),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_disponible": torch.cuda.is_available(),
        "dtype": "float32",
        "precision_matmul": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        record["gpu"] = properties.name
        record["memoria_gpu_gib"] = round(properties.total_memory / 2**30, 2)
    return record


def available_devices() -> tuple[str, ...]:
    return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


def _elapsed_ms(device: torch.device, function) -> tuple[float, object]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000, result


def benchmark_devices(
    X,
    y,
    *,
    repetitions: int = 3,
    devices: tuple[str, ...] | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    **mlp_params,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrenar el mismo MLP en cada dispositivo y comparar tiempos y resultados.

    X ya debe estar preprocesado (matriz numérica). Devuelve dos tablas: una por
    dispositivo con mediana, mínimo y máximo de `repetitions` entrenamientos, más
    pérdida inicial y final; y otra que compara los parámetros finales y las
    probabilidades de cada dispositivo con los del primero, con la tolerancia dada.
    Los tiempos son mediciones de esta sesión, no valores esperados.
    """
    if repetitions < 1:
        raise ValueError("repetitions debe ser un entero positivo.")
    devices = devices or available_devices()
    unavailable = [d for d in devices if d.startswith("cuda") and not torch.cuda.is_available()]
    if unavailable:
        raise RuntimeError(f"Dispositivos no disponibles: {unavailable}")

    X = np.asarray(X, dtype=np.float32)
    rows: list[dict] = []
    fitted: dict[str, TorchMLPClassifier] = {}
    for device in devices:
        samples: list[float] = []
        model: TorchMLPClassifier | None = None
        for _ in range(repetitions):
            model = TorchMLPClassifier(device=device, **mlp_params)
            elapsed, _ = _elapsed_ms(torch.device(device), lambda m=model: m.fit(X, y))
            samples.append(elapsed)
        assert model is not None
        fitted[device] = model
        rows.append(
            {
                "device": device,
                "epocas": model.epochs,
                "repeticiones": repetitions,
                "mediana_ms": float(np.median(samples)),
                "min_ms": float(min(samples)),
                "max_ms": float(max(samples)),
                "muestras_ms": [round(s, 3) for s in samples],
                "loss_inicial": float(model.history_["loss_antes"].iloc[0]),
                "loss_final": float(model.history_["loss_antes"].iloc[-1]),
                "loss_finita": bool(np.isfinite(model.history_["loss_antes"]).all()),
                "exactitud_final_train": float(model.history_["train_accuracy"].iloc[-1]),
            }
        )
    timings = pd.DataFrame(rows)

    to_vector = torch.nn.utils.parameters_to_vector
    reference = fitted[devices[0]]
    reference_params = to_vector(reference.network_.parameters()).detach().cpu()
    reference_proba = reference.predict_proba(X)
    comparisons: list[dict] = []
    for device, model in fitted.items():
        params = to_vector(model.network_.parameters()).detach().cpu()
        proba = model.predict_proba(X)
        comparisons.append(
            {
                "device": device,
                "referencia": devices[0],
                "parametros_cercanos": bool(
                    torch.allclose(params, reference_params, atol=atol, rtol=rtol)
                ),
                "max_diferencia_abs_parametros": float((params - reference_params).abs().max()),
                "probabilidades_cercanas": bool(
                    np.allclose(proba, reference_proba, atol=atol, rtol=rtol)
                ),
                "max_diferencia_abs_probabilidades": float(np.abs(proba - reference_proba).max()),
                "misma_clase_predicha": float(
                    (proba.argmax(1) == reference_proba.argmax(1)).mean()
                ),
                "atol": atol,
                "rtol": rtol,
            }
        )
    return timings, pd.DataFrame(comparisons)


def benchmark_transfer(model: TorchMLPClassifier, X, *, repetitions: int = 7) -> pd.DataFrame:
    """Inferencia con datos residentes en el dispositivo frente a incluir la transferencia.

    Sigue la comparación del ejercicio de la semana 3: el costo de mover datos entre
    CPU y GPU puede superar el del cálculo cuando la matriz es pequeña.
    """
    model._check_fitted()
    X = np.asarray(X, dtype=np.float32)
    device = model.device_
    resident = torch.from_numpy(X).to(device)

    def with_transfer():
        return model._logits(torch.from_numpy(X).to(device)).cpu()

    def resident_only():
        return model._logits(resident)

    rows = []
    for name, function in (("residente", resident_only), ("con_transferencias", with_transfer)):
        samples = [_elapsed_ms(device, function)[0] for _ in range(repetitions)]
        rows.append(
            {
                "device": str(device),
                "modo": name,
                "n_filas": len(X),
                "mediana_ms": float(np.median(samples)),
                "min_ms": float(min(samples)),
                "max_ms": float(max(samples)),
                "muestras_ms": [round(s, 4) for s in samples],
            }
        )
    return pd.DataFrame(rows)
