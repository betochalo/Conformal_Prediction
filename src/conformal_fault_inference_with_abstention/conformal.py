"""Cuantiles finitos y predictores split conformal y Mondrian.

El modelo se entrena una sola vez fuera de esta capa.
APS queda como extensión fuera del producto mínimo.
"""

import math
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ._validation import validate_classes
from .contracts import Labels, PredictionSets, Probabilities
from .scores import inverse_probability, true_label_scores


def conformal_rank(n: int, *, alpha: float) -> int:
    """Rango ceil((n + 1) * (1 - alpha)) del estadístico de orden usado como cuantil.

    Si el resultado supera n, el cuantil correspondiente es infinito.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar entre 0 y 1, sin incluir extremos.")
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
        raise ValueError("n debe ser un entero no negativo.")
    return math.ceil((n + 1) * (1 - alpha))


def conformal_quantile(scores: NDArray[np.float64], *, alpha: float) -> float:
    """Estadístico de orden ceil((n+1)*(1-alpha)), sin interpolación.

    Exigir 0 < alpha < 1 y puntuaciones finitas unidimensionales. Retornar infinito
    cuando el rango exceda n, también para n=0. Incluir empates con <= al predecir.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores debe ser un vector de valores finitos.")
    rank = conformal_rank(len(scores), alpha=alpha)
    if rank > len(scores):
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def _prediction_scores(
    proba: Probabilities, classes: Labels, calibrated_classes: Labels
) -> Probabilities:
    classes = validate_classes(classes)
    if not np.array_equal(classes, calibrated_classes):
        raise ValueError("El orden de classes debe coincidir con el de calibración.")
    scores = inverse_probability(proba)
    if scores.shape[1] != len(classes):
        raise ValueError("Las columnas de proba deben coincidir con classes.")
    return scores


class SplitConformal:
    """Un umbral global. Calibración y predicción reciben probabilidades."""

    def __init__(self, *, alpha: float) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha debe estar entre 0 y 1, sin incluir extremos.")
        self.alpha = alpha
        self.classes_: Labels | None = None
        self.threshold_: float | None = None

    def calibrate(self, proba: Probabilities, y: Labels, *, classes: Labels) -> Self:
        """Guardar copia de classes y cuantil global; retornar self.

        Recalibrar reemplaza el estado anterior. No entrenar el modelo aquí.
        """
        classes = validate_classes(classes)
        scores = true_label_scores(proba, y, classes=classes)
        threshold = conformal_quantile(scores, alpha=self.alpha)
        self.classes_ = classes.copy()
        self.threshold_ = threshold
        return self

    def predict_set(self, proba: Probabilities, *, classes: Labels) -> PredictionSets:
        """Aplicar <= al umbral; exigir calibración previa e idéntico orden de clases."""
        if self.classes_ is None or self.threshold_ is None:
            raise RuntimeError("Debe calibrar antes de predecir conjuntos.")
        scores = _prediction_scores(proba, classes, self.classes_)
        return PredictionSets(classes=self.classes_.copy(), mask=scores <= self.threshold_)


class MondrianConformal:
    """Un umbral por clase, usando ejemplos cuya etiqueta verdadera es esa clase."""

    def __init__(self, *, alpha: float) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha debe estar entre 0 y 1, sin incluir extremos.")
        self.alpha = alpha
        self.classes_: Labels | None = None
        self.thresholds_: NDArray[np.float64] | None = None

    def calibrate(self, proba: Probabilities, y: Labels, *, classes: Labels) -> Self:
        """Guardar copia de classes y cuantiles alineados; retornar self.

        Sin ejemplos de una clase, su umbral es infinito. Recalibrar reemplaza
        el estado anterior. No usar la clase predicha para agrupar puntuaciones.
        """
        classes = validate_classes(classes)
        scores = true_label_scores(proba, y, classes=classes)
        y = np.asarray(y)
        thresholds = np.array(
            [conformal_quantile(scores[y == label], alpha=self.alpha) for label in classes]
        )
        self.classes_ = classes.copy()
        self.thresholds_ = thresholds
        return self

    def predict_set(self, proba: Probabilities, *, classes: Labels) -> PredictionSets:
        """Usar umbral de cada candidata; validar estado y orden igual a calibración."""
        if self.classes_ is None or self.thresholds_ is None:
            raise RuntimeError("Debe calibrar antes de predecir conjuntos.")
        scores = _prediction_scores(proba, classes, self.classes_)
        return PredictionSets(classes=self.classes_.copy(), mask=scores <= self.thresholds_)
