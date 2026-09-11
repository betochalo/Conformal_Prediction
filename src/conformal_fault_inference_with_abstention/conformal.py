"""Cuantiles finitos y predictores split conformal y Mondrian.

El modelo se entrena una sola vez fuera de esta capa. Implementación pendiente.
APS queda como extensión fuera del producto mínimo.
"""

from typing import Self

import numpy as np
from numpy.typing import NDArray

from .contracts import Labels, PredictionSets, Probabilities


def conformal_quantile(scores: NDArray[np.float64], *, alpha: float) -> float:
    """Estadístico de orden ceil((n+1)*(1-alpha)), sin interpolación.

    Exigir 0 < alpha < 1 y puntuaciones finitas unidimensionales. Retornar infinito
    cuando el rango exceda n, también para n=0. Incluir empates con <= al predecir.
    """
    raise NotImplementedError("Etapa 4: cuantil con corrección finita.")


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
        raise NotImplementedError("Etapa 4: calibrar split conformal.")

    def predict_set(self, proba: Probabilities, *, classes: Labels) -> PredictionSets:
        """Aplicar <= al umbral; exigir calibración previa e idéntico orden de clases."""
        raise NotImplementedError("Etapa 4: predecir conjuntos split conformal.")


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
        raise NotImplementedError("Etapa 5: calibrar Mondrian por clase.")

    def predict_set(self, proba: Probabilities, *, classes: Labels) -> PredictionSets:
        """Usar umbral de cada candidata; validar estado y orden igual a calibración."""
        raise NotImplementedError("Etapa 5: predecir conjuntos Mondrian.")
