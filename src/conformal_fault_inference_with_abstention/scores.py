"""Función de no conformidad del producto mínimo: 1 - p(y|x).

Implementación en NumPy. APS queda como extensión.
"""

import numpy as np
from numpy.typing import NDArray

from ._validation import label_indices, validate_classes
from .contracts import Labels, Probabilities


def inverse_probability(proba: Probabilities) -> Probabilities:
    """Devolver 1-proba para todas las clases, preservando forma (n, K).

    Validar matriz finita, valores entre 0 y 1 y filas que sumen 1 con tolerancia.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] == 0:
        raise ValueError("proba debe ser una matriz con al menos una clase.")
    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("Las probabilidades deben ser finitas y estar entre 0 y 1.")
    if not np.allclose(proba.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("Cada fila de probabilidades debe sumar 1.")
    return 1.0 - proba


def true_label_scores(proba: Probabilities, y: Labels, *, classes: Labels) -> NDArray[np.float64]:
    """Seleccionar 1-proba[i, posición de y[i] en classes], sin reordenar clases.

    Validar formas compatibles, clases únicas y etiquetas conocidas.
    """
    scores = inverse_probability(proba)
    classes = validate_classes(classes)
    if scores.shape[1] != len(classes):
        raise ValueError("Las columnas de proba deben coincidir con classes.")
    indices = label_indices(y, classes, len(scores))
    return scores[np.arange(len(scores)), indices]
