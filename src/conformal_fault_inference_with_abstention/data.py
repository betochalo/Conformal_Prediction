"""Carga de AI4I, auditoría de RNF, modos simultáneos y construcción de etiquetas.

Las reglas de exclusión y prioridad se definen en LabelPolicy después de la auditoría.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import FAILURE_COLUMNS, FEATURE_COLUMNS, LabeledData, LabelPolicy

ID_COLUMN = "UID"
PRODUCT_COLUMN = "Product ID"
MACHINE_FAILURE = "Machine failure"
RNF = "RNF"
FLAG_COLUMNS = (MACHINE_FAILURE, *FAILURE_COLUMNS, RNF)
RAW_COLUMNS = (ID_COLUMN, PRODUCT_COLUMN, *FEATURE_COLUMNS, *FLAG_COLUMNS)
RNF_POLICIES = ("exclude_rows", "exclude_only_rnf")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Leer CSV local; validar columnas, IDs únicos, faltantes y flags binarios.

    Devuelve el DataFrame indexado por UID, con Product ID conservado como columna.
    Los identificadores sirven para trazabilidad, nunca como predictores.
    """
    raw = pd.read_csv(path)
    missing = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    extra = [column for column in raw.columns if column not in RAW_COLUMNS]
    if extra:
        raise ValueError(f"Columnas inesperadas en el CSV: {extra}")
    if raw.empty:
        raise ValueError("El CSV no contiene filas.")
    if raw.isna().any().any():
        cols = raw.columns[raw.isna().any()].tolist()
        raise ValueError(f"Valores faltantes en columnas: {cols}")
    if not raw[ID_COLUMN].is_unique:
        raise ValueError("UID debe ser único.")
    if not raw[PRODUCT_COLUMN].is_unique:
        raise ValueError("Product ID debe ser único.")
    for column in FLAG_COLUMNS:
        values = raw[column]
        if not pd.api.types.is_integer_dtype(values) or not values.isin((0, 1)).all():
            raise ValueError(f"La columna {column} debe ser binaria con valores 0 o 1.")
    numeric = [column for column in FEATURE_COLUMNS if column != "Type"]
    for column in numeric:
        if not pd.api.types.is_numeric_dtype(raw[column]):
            raise ValueError(f"La columna {column} debe ser numérica.")
        if not np.isfinite(raw[column].to_numpy(dtype=float)).all():
            raise ValueError(f"La columna {column} contiene valores no finitos.")
    return raw.set_index(ID_COLUMN)


def _physical_mask(raw: pd.DataFrame) -> pd.Series:
    return raw[list(FAILURE_COLUMNS)].sum(axis=1) >= 1


def _check_flags(raw: pd.DataFrame) -> None:
    missing = [column for column in FLAG_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas de fallas: {missing}")
    for column in FLAG_COLUMNS:
        if not raw[column].isin((0, 1)).all():
            raise ValueError(f"La columna {column} debe ser binaria con valores 0 o 1.")


def audit_failures(raw: pd.DataFrame) -> pd.DataFrame:
    """Tabla metric/count: modos, RNF, solapamientos e inconsistencias.

    Incluye RNF solo, modos físicos simultáneos, Machine failure sin modo y
    modos activos con Machine failure=0. No modifica ni excluye filas.
    """
    _check_flags(raw)
    failure = raw[MACHINE_FAILURE] == 1
    rnf = raw[RNF] == 1
    n_physical = raw[list(FAILURE_COLUMNS)].sum(axis=1)
    physical = n_physical >= 1

    rows: list[tuple[str, int]] = [("rows", len(raw)), ("machine_failure", int(failure.sum()))]
    rows += [(f"flag_{column}", int(raw[column].sum())) for column in (*FAILURE_COLUMNS, RNF)]
    rows += [
        ("rows_with_physical_mode", int(physical.sum())),
        ("rows_with_multiple_physical_modes", int((n_physical >= 2).sum())),
    ]
    for size in (2, 3, 4):
        for combo in combinations(FAILURE_COLUMNS, size):
            active = (raw[list(combo)] == 1).all(axis=1) & (n_physical == size)
            count = int(active.sum())
            if count:
                rows.append((f"simultaneous_{'+'.join(combo)}", count))
    rows += [
        ("rnf_only", int((rnf & ~physical).sum())),
        ("rnf_with_physical_mode", int((rnf & physical).sum())),
        ("machine_failure_without_mode", int((failure & ~physical & ~rnf).sum())),
        ("machine_failure_only_rnf", int((failure & ~physical & rnf).sum())),
        ("mode_without_machine_failure", int((~failure & physical).sum())),
        ("rnf_without_machine_failure", int((~failure & rnf).sum())),
        ("normal_strict", int((~failure & ~physical & ~rnf).sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "count"])


def _validate_policy(policy: LabelPolicy) -> None:
    if tuple(sorted(policy.priority)) != tuple(sorted(FAILURE_COLUMNS)):
        raise ValueError(f"priority debe ser una permutación de {FAILURE_COLUMNS}.")
    if policy.rnf_policy not in RNF_POLICIES:
        raise ValueError(f"rnf_policy debe ser una de {RNF_POLICIES}.")


def exclusion_mask(raw: pd.DataFrame, *, policy: LabelPolicy) -> pd.DataFrame:
    """Motivos de exclusión por fila: failure_without_mode y rnf, según la política.

    Una fila puede activar más de un motivo. Sirve para reportar exclusiones sin
    volver a calcular las reglas fuera de este módulo.
    """
    _validate_policy(policy)
    _check_flags(raw)
    failure = raw[MACHINE_FAILURE] == 1
    rnf = raw[RNF] == 1
    physical = _physical_mask(raw)
    without_mode = failure & ~physical & ~rnf
    if policy.rnf_policy == "exclude_rows":
        rnf_excluded = rnf
    else:
        rnf_excluded = rnf & ~physical
    return pd.DataFrame(
        {"failure_without_mode": without_mode, "rnf": rnf_excluded}, index=raw.index
    )


def build_labels(raw: pd.DataFrame, *, policy: LabelPolicy) -> LabeledData:
    """Aplicar política explícita; devolver solo FEATURE_COLUMNS y etiquetas.

    Normal exige Machine failure=0 y todos los modos, incluido RNF, apagados.
    Ante modo físico activo, usa priority aunque Machine failure sea inconsistente;
    la auditoría debe haber reportado esos casos. Conserva el índice de origen.
    """
    _validate_policy(policy)
    _check_flags(raw)
    missing = [column for column in FEATURE_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas predictoras: {missing}")

    physical = _physical_mask(raw)
    excluded = exclusion_mask(raw, policy=policy).any(axis=1)

    labels = pd.Series("Normal", index=raw.index, dtype=object)
    for mode in reversed(policy.priority):
        labels[raw[mode] == 1] = mode
    labels[~physical & (raw[MACHINE_FAILURE] == 1)] = "__no_mode__"
    labels[~physical & (raw[RNF] == 1)] = "__rnf__"

    keep = ~excluded
    unresolved = labels[keep].isin(("__no_mode__", "__rnf__"))
    if unresolved.any():
        raise RuntimeError("Quedaron filas sin etiqueta válida tras aplicar la política.")

    X = raw.loc[keep, list(FEATURE_COLUMNS)].copy()
    y = labels[keep].to_numpy(dtype=np.str_)
    return LabeledData(X=X, y=y)
