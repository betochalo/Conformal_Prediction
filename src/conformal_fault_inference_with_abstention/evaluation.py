"""Cobertura marginal y por clase, tamaño medio del conjunto y abstención.

Reportar los conteos que sustentan las métricas. Implementación pendiente.
El costo empírico y las clases no vistas quedan como extensiones.
"""

from .contracts import EvaluationReport, Labels, PredictionSets


def evaluate_sets(y_true: Labels, prediction_sets: PredictionSets) -> EvaluationReport:
    """Medir pertenencia de la etiqueta verdadera, tamaño y fracciones de rutas.

    Validar filas alineadas, clases únicas y etiquetas conocidas; rechazar prueba
    vacía. Incluir todas las clases en by_class, aun si support=0 (métricas NaN).
    Abstención = assisted_rate + human_rate = fracción de tamaños distintos de 1.
    """
    raise NotImplementedError("Etapa 6: métricas globales y por clase.")
