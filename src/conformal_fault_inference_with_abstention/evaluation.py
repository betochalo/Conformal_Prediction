"""Cobertura marginal y por clase, tamaño medio del conjunto, abstención y errores automáticos.

Reporta los conteos que sustentan las métricas. La cobertura conforme no garantiza
la exactitud condicionada a decidir de forma automática, así que esa exactitud se
mide aparte (clasificación selectiva). El costo empírico y las clases no vistas
quedan como extensiones.
"""

import numpy as np
import pandas as pd

from ._validation import label_indices, validate_sets
from .contracts import EvaluationReport, Labels, PredictionSets
from .decision import route_predictions


def automatic_decision_errors(y_true: Labels, prediction_sets: PredictionSets) -> pd.DataFrame:
    """Errores entre las filas con conjunto de tamaño 1, por clase verdadera.

    Columnas: class, automatic (decisiones automáticas con esa clase verdadera),
    errors (predicción distinta de la verdadera), error_rate (NaN sin decisiones)
    y predicted_as (clase predicha más frecuente entre los errores, vacío si no hay).
    Incluye todas las clases del conjunto de predicción, aun sin casos.
    """
    classes, mask = validate_sets(prediction_sets)
    indices = label_indices(y_true, classes, len(mask))
    automatic = mask.sum(axis=1) == 1
    predicted = mask.argmax(axis=1)
    wrong = automatic & (predicted != indices)
    rows = []
    for index, label in enumerate(classes):
        selected = indices == index
        n_auto = int((automatic & selected).sum())
        errors = wrong & selected
        n_err = int(errors.sum())
        if n_err:
            counts = np.bincount(predicted[errors], minlength=len(classes))
            predicted_as = str(classes[int(counts.argmax())])
        else:
            predicted_as = ""
        rows.append(
            {
                "class": label,
                "automatic": n_auto,
                "errors": n_err,
                "error_rate": n_err / n_auto if n_auto else np.nan,
                "predicted_as": predicted_as,
            }
        )
    return pd.DataFrame(rows)


def evaluate_sets(y_true: Labels, prediction_sets: PredictionSets) -> EvaluationReport:
    """Medir pertenencia de la etiqueta verdadera, tamaño, fracciones de rutas y errores.

    Validar filas alineadas, clases únicas y etiquetas conocidas; rechazar prueba
    vacía. Incluir todas las clases en by_class, aun si support=0 (métricas NaN).
    Abstención = assisted_rate + human_rate = fracción de tamaños distintos de 1.
    automatic_error_rate es la fracción de decisiones automáticas incorrectas; NaN
    si no hay ninguna decisión automática.
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
    errors = automatic_decision_errors(y_true, prediction_sets)
    automatic_count = int(errors["automatic"].sum())
    automatic_errors = int(errors["errors"].sum())
    return EvaluationReport(
        n_samples=len(mask),
        marginal_coverage=float(covered.mean()),
        mean_set_size=float(sizes.mean()),
        abstention_rate=float(abstained.mean()),
        assisted_rate=float((actions == "assisted").mean()),
        human_rate=float((actions == "human").mean()),
        automatic_count=automatic_count,
        automatic_errors=automatic_errors,
        automatic_error_rate=automatic_errors / automatic_count if automatic_count else np.nan,
        by_class=pd.DataFrame(rows),
        automatic_errors_by_class=errors,
    )
