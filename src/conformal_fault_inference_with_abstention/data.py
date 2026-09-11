"""Carga de AI4I, auditoría de RNF, modos simultáneos y construcción de etiquetas.

Las reglas de exclusión y prioridad deben definirse antes de implementarlas.
"""

from pathlib import Path

import pandas as pd

from .contracts import LabeledData, LabelPolicy


def load_raw(path: str | Path) -> pd.DataFrame:
    """Leer CSV local; validar columnas, IDs únicos, faltantes y flags binarios.

    Conservar los identificadores para trazabilidad, nunca como predictores.
    """
    raise NotImplementedError("Etapa 1: carga y validación de AI4I.")


def audit_failures(raw: pd.DataFrame) -> pd.DataFrame:
    """Tabla metric/count: modos, RNF, solapamientos e inconsistencias.

    Incluir RNF solo, modos físicos simultáneos, Machine failure sin modo y
    modos activos con Machine failure=0. No modificar ni excluir filas aquí.
    """
    raise NotImplementedError("Etapa 1: auditoría antes de decidir las etiquetas.")


def build_labels(raw: pd.DataFrame, *, policy: LabelPolicy) -> LabeledData:
    """Aplicar política explícita; devolver solo FEATURE_COLUMNS y etiquetas.

    Normal exige Machine failure=0 y todos los modos, incluido RNF, apagados.
    Ante modo físico activo, usar priority aunque Machine failure sea inconsistente;
    la auditoría debe haber reportado esos casos. Conservar el índice de origen.
    Validar priority como permutación completa y rnf_policy como opción admitida.
    """
    raise NotImplementedError("Etapa 1: acordar y aplicar la política de etiquetas.")
