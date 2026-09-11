"""Abstención cuando el conjunto no contiene exactamente una etiqueta.

Desglose: asistida con dos etiquetas, humana con cero o más de dos.
La política sensible al costo queda como extensión.
"""

import numpy as np

from ._validation import validate_sets
from .contracts import Actions, PredictionSets


def route_predictions(prediction_sets: PredictionSets) -> Actions:
    """Por fila: 'automatic' si tamaño=1, 'assisted' si =2, 'human' si =0 o >2."""
    _, mask = validate_sets(prediction_sets)
    sizes = mask.sum(axis=1)
    return np.where(sizes == 1, "automatic", np.where(sizes == 2, "assisted", "human"))
