"""Pruebas de carga, auditoría y etiquetas con filas sintéticas pequeñas."""

import numpy as np
import pandas as pd
import pytest

from conformal_fault_inference_with_abstention.contracts import (
    FAILURE_COLUMNS,
    FEATURE_COLUMNS,
    LabelPolicy,
)
from conformal_fault_inference_with_abstention.data import (
    audit_failures,
    build_labels,
    exclusion_mask,
    load_raw,
)

POLICY = LabelPolicy(priority=("TWF", "PWF", "OSF", "HDF"), rnf_policy="exclude_only_rnf")


def make_rows(flags: list[dict[str, int]]) -> pd.DataFrame:
    """Construir un DataFrame con predictores constantes y flags dados por fila."""
    n = len(flags)
    frame = pd.DataFrame(
        {
            "UID": np.arange(1, n + 1),
            "Product ID": [f"L{i:05d}" for i in range(n)],
            "Type": ["L"] * n,
            "Air temperature": np.full(n, 298.1),
            "Process temperature": np.full(n, 308.6),
            "Rotational speed": np.full(n, 1551),
            "Torque": np.full(n, 42.8),
            "Tool wear": np.arange(n),
        }
    )
    for column in ("Machine failure", *FAILURE_COLUMNS, "RNF"):
        frame[column] = [row.get(column, 0) for row in flags]
    return frame


# Filas de referencia: cada caso relevante para la auditoría y la política.
ROWS = [
    {},  # 0: Normal
    {"Machine failure": 1, "TWF": 1},  # 1: TWF simple
    {"Machine failure": 1, "HDF": 1, "OSF": 1},  # 2: solapamiento HDF+OSF
    {"Machine failure": 1, "TWF": 1, "PWF": 1, "OSF": 1},  # 3: triple
    {"RNF": 1},  # 4: RNF solo, Machine failure=0
    {"Machine failure": 1, "RNF": 1},  # 5: RNF solo, Machine failure=1
    {"Machine failure": 1, "PWF": 1, "RNF": 1},  # 6: RNF con modo físico
    {"Machine failure": 1},  # 7: falla sin modo
    {"HDF": 1},  # 8: modo activo con Machine failure=0
]


def test_load_raw_reads_and_indexes_by_uid(tmp_path):
    path = tmp_path / "ai4i.csv"
    make_rows(ROWS).to_csv(path, index=False)
    raw = load_raw(path)
    assert raw.index.name == "UID"
    assert list(raw.index) == list(range(1, len(ROWS) + 1))
    assert "Product ID" in raw.columns
    assert all(column in raw.columns for column in FEATURE_COLUMNS)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda df: df.drop(columns=["Torque"]), "Faltan columnas"),
        (lambda df: df.assign(extra=1), "inesperadas"),
        (lambda df: df.assign(UID=[1] * len(df)), "UID debe ser único"),
        (lambda df: df.assign(TWF=[2] + [0] * (len(df) - 1)), "binaria"),
        (lambda df: df.assign(Torque=[np.nan] + [1.0] * (len(df) - 1)), "faltantes"),
        (lambda df: df.iloc[:0], "no contiene filas"),
    ],
)
def test_load_raw_rejects_invalid_files(tmp_path, mutate, message):
    path = tmp_path / "bad.csv"
    mutate(make_rows(ROWS)).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_raw(path)


def test_audit_counts_indicators_overlaps_and_inconsistencies():
    audit = audit_failures(make_rows(ROWS)).set_index("metric")["count"]
    assert audit["rows"] == 9
    assert audit["machine_failure"] == 6
    assert audit["flag_TWF"] == 2
    assert audit["flag_HDF"] == 2
    assert audit["flag_PWF"] == 2
    assert audit["flag_OSF"] == 2
    assert audit["flag_RNF"] == 3
    assert audit["rows_with_physical_mode"] == 5
    assert audit["rows_with_multiple_physical_modes"] == 2
    assert audit["simultaneous_HDF+OSF"] == 1
    assert audit["simultaneous_TWF+PWF+OSF"] == 1
    assert "simultaneous_TWF+HDF" not in audit.index
    assert audit["rnf_only"] == 2
    assert audit["rnf_with_physical_mode"] == 1
    assert audit["machine_failure_without_mode"] == 1
    assert audit["machine_failure_only_rnf"] == 1
    assert audit["mode_without_machine_failure"] == 1
    assert audit["rnf_without_machine_failure"] == 1
    assert audit["normal_strict"] == 1


def test_audit_does_not_modify_input():
    raw = make_rows(ROWS)
    before = raw.copy()
    audit_failures(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_build_labels_applies_priority_and_keeps_only_features():
    raw = make_rows(ROWS).set_index("UID")
    data = build_labels(raw, policy=POLICY)
    assert list(data.X.columns) == list(FEATURE_COLUMNS)
    assert data.X.index.name == "UID"
    # Excluidos: 4 y 5 (RNF sin modo), 7 (falla sin modo). Se conservan 0,1,2,3,6,8.
    assert list(data.X.index) == [1, 2, 3, 4, 7, 9]
    assert data.y.tolist() == ["Normal", "TWF", "OSF", "TWF", "PWF", "HDF"]
    assert data.y.dtype.kind == "U"
    assert len(data.y) == len(data.X)


def test_build_labels_priority_order_changes_overlaps():
    raw = make_rows(ROWS).set_index("UID")
    policy = LabelPolicy(priority=("HDF", "OSF", "PWF", "TWF"), rnf_policy="exclude_only_rnf")
    data = build_labels(raw, policy=policy)
    assert data.y.tolist() == ["Normal", "TWF", "HDF", "OSF", "PWF", "HDF"]


def test_exclude_rows_policy_drops_rnf_with_physical_mode():
    raw = make_rows(ROWS).set_index("UID")
    policy = LabelPolicy(priority=POLICY.priority, rnf_policy="exclude_rows")
    data = build_labels(raw, policy=policy)
    assert list(data.X.index) == [1, 2, 3, 4, 9]
    assert data.y.tolist() == ["Normal", "TWF", "OSF", "TWF", "HDF"]


def test_exclusion_mask_reports_reasons_per_row():
    raw = make_rows(ROWS).set_index("UID")
    mask = exclusion_mask(raw, policy=POLICY)
    assert mask["failure_without_mode"].sum() == 1
    assert mask.loc[8, "failure_without_mode"]
    assert mask["rnf"].sum() == 2
    assert mask.loc[[5, 6], "rnf"].all()
    assert not mask.loc[7, "rnf"]
    strict = exclusion_mask(raw, policy=LabelPolicy(POLICY.priority, "exclude_rows"))
    assert strict["rnf"].sum() == 3


def test_normal_requires_all_flags_off():
    raw = make_rows([{}, {"RNF": 1}, {"Machine failure": 1}]).set_index("UID")
    data = build_labels(raw, policy=POLICY)
    assert data.y.tolist() == ["Normal"]
    assert list(data.X.index) == [1]


@pytest.mark.parametrize(
    "policy",
    [
        LabelPolicy(priority=("TWF", "HDF", "PWF"), rnf_policy="exclude_rows"),
        LabelPolicy(priority=("TWF", "HDF", "PWF", "PWF"), rnf_policy="exclude_rows"),
        LabelPolicy(priority=("TWF", "HDF", "PWF", "OSF", "RNF"), rnf_policy="exclude_rows"),
        LabelPolicy(priority=("TWF", "HDF", "PWF", "OSF"), rnf_policy="keep"),  # type: ignore[arg-type]
    ],
)
def test_build_labels_rejects_invalid_policy(policy):
    with pytest.raises(ValueError):
        build_labels(make_rows(ROWS), policy=policy)


def test_build_labels_rejects_non_binary_flags():
    raw = make_rows(ROWS)
    raw.loc[0, "HDF"] = 3
    with pytest.raises(ValueError, match="binaria"):
        build_labels(raw, policy=POLICY)
