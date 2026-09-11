"""Validaciones compartidas de clases, etiquetas y conjuntos."""

import numpy as np

from .contracts import Labels, PredictionSets, SetMask


def validate_classes(classes: Labels) -> Labels:
    classes = np.asarray(classes)
    if classes.ndim != 1 or classes.size == 0:
        raise ValueError("classes debe ser un vector no vacío.")
    if not all(isinstance(value, str) and value for value in classes):
        raise ValueError("classes debe contener nombres de clase no vacíos.")
    if len(set(classes)) != len(classes):
        raise ValueError("classes debe contener clases únicas.")
    return classes.astype(np.str_)


def label_indices(y: Labels, classes: Labels, n_rows: int) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n_rows:
        raise ValueError("Las etiquetas deben ser un vector alineado con las filas.")
    positions = {label: i for i, label in enumerate(classes)}
    if any(not isinstance(label, str) or label not in positions for label in y):
        raise ValueError("Las etiquetas contienen clases desconocidas.")
    return np.array([positions[label] for label in y], dtype=np.intp)


def validate_sets(prediction_sets: PredictionSets) -> tuple[Labels, SetMask]:
    classes = validate_classes(prediction_sets.classes)
    mask = np.asarray(prediction_sets.mask)
    if mask.ndim != 2 or mask.shape[1] != len(classes):
        raise ValueError("mask debe tener forma (n_observaciones, n_clases).")
    if mask.dtype != np.bool_:
        raise ValueError("mask debe contener booleanos.")
    return classes, mask
