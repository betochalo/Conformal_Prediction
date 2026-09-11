"""Abstención cuando el conjunto no contiene exactamente una etiqueta.

Desglose: asistida con dos etiquetas, humana con cero o más de dos.
Implementación pendiente. La política sensible al costo queda como extensión.
"""

from .contracts import Actions, PredictionSets


def route_predictions(prediction_sets: PredictionSets) -> Actions:
    """Por fila: 'automatic' si tamaño=1, 'assisted' si =2, 'human' si =0 o >2."""
    raise NotImplementedError("Etapa 6: enrutamiento por tamaño del conjunto.")
