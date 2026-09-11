"""Particiones y reporte obligatorio previo al entrenamiento."""

import pandas as pd

from .contracts import DataSplits, LabeledData


def split_data(
    data: LabeledData,
    *,
    calibration_size: float,
    test_size: float,
    random_state: int,
) -> DataSplits:
    """Separar con estratificación; tamaños relativos al total, no al remanente.

    Validar fracciones positivas cuya suma sea menor que 1, disjunción y presencia
    de clases en entrenamiento. Fallar explícitamente si la partición es inviable.
    No sobremuestrear calibración ni prueba. Guardar los índices para reproducir.
    """
    raise NotImplementedError("Etapa 2: particiones reproducibles.")


def class_counts(splits: DataSplits) -> pd.DataFrame:
    """Filas CLASSES; columnas train, calibration, test, total; incluir ceros."""
    raise NotImplementedError("Etapa 2: tabla obligatoria de conteos.")


def calibration_feasibility(splits: DataSplits, *, alpha: float) -> pd.DataFrame:
    """Por clase: n_calibration, rank y finite_threshold_possible.

    rank = ceil((n_calibration + 1) * (1 - alpha)). Marcar cuando exceda la
    muestra, incluido n=0. Esto no mide precisión estadística ni intercambiabilidad.
    """
    raise NotImplementedError("Etapa 2: viabilidad de calibración Mondrian.")
