"""Tipos compartidos para que los módulos puedan desarrollarse por separado."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type Labels = NDArray[np.str_]
type Probabilities = NDArray[np.float64]
type SetMask = NDArray[np.bool_]
type Actions = NDArray[np.str_]
type RNFPolicy = Literal["exclude_rows", "exclude_only_rnf"]

CLASSES = ("Normal", "TWF", "HDF", "PWF", "OSF")
FEATURE_COLUMNS = (
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
)
FAILURE_COLUMNS = ("TWF", "HDF", "PWF", "OSF")


@dataclass(frozen=True)
class LabelPolicy:
    """Decisiones que se deben justificar después de la auditoría.

    priority: permutación de los cuatro modos; el primero activo gana.
    exclude_rows: excluir cualquier registro RNF.
    exclude_only_rnf: excluir RNF sin modo físico y conservar los demás.
    Los fallos sin modo identificado se excluyen y se reportan en ambos casos.
    """

    priority: tuple[str, ...]
    rnf_policy: RNFPolicy


@dataclass(frozen=True)
class LabeledData:
    """X e y alineados, con índice original único conservado en X."""

    X: pd.DataFrame
    y: Labels


@dataclass(frozen=True)
class DataSplits:
    """Bloques disjuntos, con las mismas columnas y convención de etiquetas."""

    train: LabeledData
    calibration: LabeledData
    test: LabeledData


@dataclass(frozen=True)
class PredictionSets:
    """mask[i, j] indica si classes[j] pertenece al conjunto de la fila i.

    mask tiene forma (n_observaciones, n_clases), incluso con conjuntos vacíos.
    El orden de classes debe coincidir con las columnas de predict_proba.
    """

    classes: Labels
    mask: SetMask


@dataclass(frozen=True)
class EvaluationReport:
    """Cobertura no estimable: NaN y support=0 en la tabla por clase.

    by_class: columnas class, support, covered, coverage, mean_set_size,
    abstention_rate. abstention_rate incluye las vías asistida y humana.
    """

    n_samples: int
    marginal_coverage: float
    mean_set_size: float
    abstention_rate: float
    assisted_rate: float
    human_rate: float
    by_class: pd.DataFrame
