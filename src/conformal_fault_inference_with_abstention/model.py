"""Único modelo base del producto mínimo y su preprocesamiento."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .contracts import FEATURE_COLUMNS


def build_model(*, random_state: int) -> Pipeline:
    """Construir, sin entrenar, preprocesamiento de Type y un HistGradientBoosting.

    Usar variables originales. Codificar Type con tratamiento explícito de valores
    desconocidos; mantener salida densa. No hace falta escalar para árboles.
    Todo ajuste ocurre al invocar fit únicamente con entrenamiento.
    El modelo final debe exponer classes_ y predict_proba en ese mismo orden.
    """
    preprocessing = ColumnTransformer(
        [
            ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Type"]),
            ("numeric", "passthrough", list(FEATURE_COLUMNS[1:])),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                HistGradientBoostingClassifier(random_state=random_state, early_stopping=False),
            ),
        ]
    )
