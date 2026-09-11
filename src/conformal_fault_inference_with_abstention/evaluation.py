"""Cobertura marginal y por clase, tamaño medio del conjunto y abstención.

Reporta los conteos que sustentan las métricas.
El costo empírico y las clases no vistas quedan como extensiones.
"""

import numpy as np
import pandas as pd

from ._validation import label_indices, validate_sets
from .contracts import EvaluationReport, Labels, PredictionSets
from .decision import route_predictions


def evaluate_sets(y_true: Labels, prediction_sets: PredictionSets) -> EvaluationReport:
    """Medir pertenencia de la etiqueta verdadera, tamaño y fracciones de rutas.

    Validar filas alineadas, clases únicas y etiquetas conocidas; rechazar prueba
    vacía. Incluir todas las clases en by_class, aun si support=0 (métricas NaN).
    Abstención = assisted_rate + human_rate = fracción de tamaños distintos de 1.
    """
    classes, mask = validate_sets(prediction_sets)
    if len(mask) == 0:
        raise ValueError("No se puede evaluar una prueba vacía.")
    indices = label_indices(y_true, classes, len(mask))
    covered = mask[np.arange(len(mask)), indices]
    sizes = mask.sum(axis=1)
    abstained = sizes != 1
    actions = route_predictions(prediction_sets)
    rows = []
    for index, label in enumerate(classes):
        selected = indices == index
        support = int(selected.sum())
        rows.append(
            {
                "class": label,
                "support": support,
                "covered": int(covered[selected].sum()),
                "coverage": float(covered[selected].mean()) if support else np.nan,
                "mean_set_size": float(sizes[selected].mean()) if support else np.nan,
                "abstention_rate": float(abstained[selected].mean()) if support else np.nan,
            }
        )
    return EvaluationReport(
        n_samples=len(mask),
        marginal_coverage=float(covered.mean()),
        mean_set_size=float(sizes.mean()),
        abstention_rate=float(abstained.mean()),
        assisted_rate=float((actions == "assisted").mean()),
        human_rate=float((actions == "human").mean()),
        by_class=pd.DataFrame(rows),
    )
