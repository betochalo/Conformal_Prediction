"""Particiones y reporte obligatorio previo al entrenamiento."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .conformal import conformal_rank
from .contracts import CLASSES, DataSplits, LabeledData


def _validate_labeled(data: LabeledData) -> None:
    if len(data.X) != len(data.y):
        raise ValueError("X e y deben tener la misma cantidad de filas.")
    if len(data.X) == 0:
        raise ValueError("No hay filas para particionar.")
    if not data.X.index.is_unique:
        raise ValueError("El índice de X debe ser único para rastrear las particiones.")
    unknown = set(np.unique(data.y)) - set(CLASSES)
    if unknown:
        raise ValueError(f"Etiquetas fuera de CLASSES: {sorted(unknown)}")


def split_data(
    data: LabeledData,
    *,
    calibration_size: float,
    test_size: float,
    random_state: int,
) -> DataSplits:
    """Separar con estratificación; tamaños relativos al total, no al remanente.

    Primero se aparta prueba y luego calibración del remanente, reescalando la
    fracción para que ambas queden referidas al total. Falla si alguna fracción no
    es positiva, si su suma no es menor que 1, o si alguna clase presente queda
    fuera de entrenamiento. Nunca sobremuestrea: cada fila aparece exactamente una vez.
    """
    _validate_labeled(data)
    if not (0 < calibration_size < 1 and 0 < test_size < 1):
        raise ValueError("calibration_size y test_size deben estar en (0, 1).")
    if calibration_size + test_size >= 1:
        raise ValueError("calibration_size + test_size debe ser menor que 1.")

    counts = pd.Series(data.y).value_counts()
    too_small = counts[counts < 3]
    if not too_small.empty:
        raise ValueError(
            "Cada clase necesita al menos 3 filas para repartirse en tres bloques: "
            f"{too_small.to_dict()}"
        )

    index = data.X.index.to_numpy()
    y = np.asarray(data.y)
    rest_idx, test_idx, rest_y, _ = train_test_split(
        index, y, test_size=test_size, stratify=y, random_state=random_state
    )
    calibration_fraction = calibration_size / (1 - test_size)
    train_idx, cal_idx = train_test_split(
        rest_idx, test_size=calibration_fraction, stratify=rest_y, random_state=random_state
    )

    blocks = [np.sort(block) for block in (train_idx, cal_idx, test_idx)]
    total = np.concatenate(blocks)
    if len(total) != len(index) or len(np.unique(total)) != len(index):
        raise RuntimeError("Las particiones no conservan exactamente todas las filas.")

    y_series = pd.Series(y, index=data.X.index)

    def take(block: np.ndarray) -> LabeledData:
        labels = y_series.loc[block].to_numpy(dtype=np.str_)
        return LabeledData(X=data.X.loc[block].copy(), y=labels)

    splits = DataSplits(train=take(blocks[0]), calibration=take(blocks[1]), test=take(blocks[2]))
    present = set(counts.index)
    missing_train = present - set(np.unique(splits.train.y))
    if missing_train:
        raise ValueError(f"Clases ausentes en entrenamiento: {sorted(missing_train)}")
    return splits


def class_counts(splits: DataSplits) -> pd.DataFrame:
    """Filas CLASSES; columnas train, calibration, test, total; incluir ceros."""
    table = pd.DataFrame(index=pd.Index(CLASSES, name="class"))
    for name in ("train", "calibration", "test"):
        block: LabeledData = getattr(splits, name)
        counts = pd.Series(block.y).value_counts()
        table[name] = counts.reindex(CLASSES, fill_value=0).astype(int)
    table["total"] = table[["train", "calibration", "test"]].sum(axis=1)
    return table


def calibration_feasibility(splits: DataSplits, *, alpha: float) -> pd.DataFrame:
    """Por clase: n_calibration, rank y finite_threshold_possible.

    rank = ceil((n_calibration + 1) * (1 - alpha)). finite_threshold_possible es
    falso cuando el rango excede la muestra, incluido n=0. Esto no mide precisión
    estadística ni intercambiabilidad.
    """
    counts = class_counts(splits)["calibration"]
    table = pd.DataFrame(index=counts.index)
    table["alpha"] = alpha
    table["n_calibration"] = counts.astype(int)
    table["rank"] = [conformal_rank(int(n), alpha=alpha) for n in counts]
    table["finite_threshold_possible"] = table["rank"] <= table["n_calibration"]
    return table
