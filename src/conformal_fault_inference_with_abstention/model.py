"""Único modelo base del producto mínimo y su preprocesamiento."""

from sklearn.pipeline import Pipeline


def build_model(*, random_state: int) -> Pipeline:
    """Construir, sin entrenar, preprocesamiento de Type y un HistGradientBoosting.

    Usar variables originales. Codificar Type con tratamiento explícito de valores
    desconocidos; mantener salida densa. No hace falta escalar para árboles.
    Todo ajuste ocurre al invocar fit únicamente con entrenamiento.
    El modelo final debe exponer classes_ y predict_proba en ese mismo orden.
    """
    raise NotImplementedError("Etapa 3: construir el único clasificador base.")
