"""Función de no conformidad del producto mínimo: 1 - p(y|x).

Implementación pendiente en NumPy. APS queda como extensión.
"""

import numpy as np
from numpy.typing import NDArray

from .contracts import Labels, Probabilities


def inverse_probability(proba: Probabilities) -> Probabilities:
    """Devolver 1-proba para todas las clases, preservando forma (n, K).

    Validar matriz finita, valores entre 0 y 1 y filas que sumen 1 con tolerancia.
    """
    raise NotImplementedError("Etapa 4: puntuaciones candidatas.")


def true_label_scores(proba: Probabilities, y: Labels, *, classes: Labels) -> NDArray[np.float64]:
    """Seleccionar 1-proba[i, posición de y[i] en classes], sin reordenar clases.

    Validar formas compatibles, clases únicas y etiquetas conocidas.
    """
    raise NotImplementedError("Etapa 4: puntuaciones de calibración.")
