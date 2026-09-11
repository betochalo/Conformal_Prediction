"""Pruebas de particiones, conteos y viabilidad de calibración con datos sintéticos."""

import numpy as np
import pandas as pd
import pytest

from conformal_fault_inference_with_abstention.conformal import conformal_rank
from conformal_fault_inference_with_abstention.contracts import (
    CLASSES,
    FEATURE_COLUMNS,
    DataSplits,
    LabeledData,
)
from conformal_fault_inference_with_abstention.splitting import (
    calibration_feasibility,
    class_counts,
    split_data,
)


def make_data(counts: dict[str, int], *, start_uid: int = 1) -> LabeledData:
    labels = [label for label, n in counts.items() for _ in range(n)]
    n = len(labels)
    X = pd.DataFrame(
        {
            "Type": ["L"] * n,
            "Air temperature": np.linspace(295.0, 305.0, n),
            "Process temperature": np.linspace(305.0, 315.0, n),
            "Rotational speed": np.arange(n) + 1200,
            "Torque": np.linspace(10.0, 70.0, n),
            "Tool wear": np.arange(n),
        },
        index=pd.Index(np.arange(start_uid, start_uid + n), name="UID"),
    )
    return LabeledData(X=X, y=np.array(labels, dtype=np.str_))


COUNTS = {"Normal": 200, "TWF": 10, "HDF": 25, "PWF": 20, "OSF": 15}


def test_split_is_disjoint_and_preserves_all_rows():
    data = make_data(COUNTS)
    splits = split_data(data, calibration_size=0.3, test_size=0.2, random_state=0)
    blocks = [splits.train.X.index, splits.calibration.X.index, splits.test.X.index]
    union = blocks[0].union(blocks[1]).union(blocks[2])
    assert len(union) == len(data.X)
    assert union.equals(data.X.index.sort_values())
    assert blocks[0].intersection(blocks[1]).empty
    assert blocks[0].intersection(blocks[2]).empty
    assert blocks[1].intersection(blocks[2]).empty
    for block in (splits.train, splits.calibration, splits.test):
        assert list(block.X.columns) == list(FEATURE_COLUMNS)
        assert len(block.X) == len(block.y)
        assert block.y.dtype.kind == "U"


def test_labels_stay_aligned_with_rows():
    data = make_data(COUNTS)
    splits = split_data(data, calibration_size=0.3, test_size=0.2, random_state=1)
    original = pd.Series(data.y, index=data.X.index)
    for block in (splits.train, splits.calibration, splits.test):
        assert (original.loc[block.X.index].to_numpy() == block.y).all()


def test_fractions_are_relative_to_total():
    data = make_data(COUNTS)
    splits = split_data(data, calibration_size=0.3, test_size=0.2, random_state=0)
    n = len(data.X)
    assert len(splits.test.X) == pytest.approx(0.2 * n, abs=2)
    assert len(splits.calibration.X) == pytest.approx(0.3 * n, abs=2)
    assert len(splits.train.X) == pytest.approx(0.5 * n, abs=2)


def test_split_is_stratified():
    data = make_data(COUNTS)
    splits = split_data(data, calibration_size=0.3, test_size=0.2, random_state=0)
    table = class_counts(splits)
    for label, total in COUNTS.items():
        assert table.loc[label, "total"] == total
        assert table.loc[label, "test"] == pytest.approx(0.2 * total, abs=1)
        assert table.loc[label, "calibration"] == pytest.approx(0.3 * total, abs=1)


def test_split_is_reproducible_with_seed():
    data = make_data(COUNTS)
    a = split_data(data, calibration_size=0.3, test_size=0.2, random_state=7)
    b = split_data(data, calibration_size=0.3, test_size=0.2, random_state=7)
    c = split_data(data, calibration_size=0.3, test_size=0.2, random_state=8)
    assert a.train.X.index.equals(b.train.X.index)
    assert a.calibration.X.index.equals(b.calibration.X.index)
    assert a.test.X.index.equals(b.test.X.index)
    assert not a.train.X.index.equals(c.train.X.index)


@pytest.mark.parametrize(
    "calibration_size, test_size",
    [(0.0, 0.2), (0.3, 0.0), (0.5, 0.5), (0.7, 0.4), (-0.1, 0.2), (0.3, 1.0)],
)
def test_split_rejects_invalid_fractions(calibration_size, test_size):
    with pytest.raises(ValueError):
        split_data(
            make_data(COUNTS),
            calibration_size=calibration_size,
            test_size=test_size,
            random_state=0,
        )


def test_split_rejects_classes_too_small_to_split():
    data = make_data({"Normal": 50, "TWF": 2})
    with pytest.raises(ValueError, match="al menos 3"):
        split_data(data, calibration_size=0.3, test_size=0.2, random_state=0)


def test_split_rejects_unknown_labels_and_duplicate_index():
    data = make_data(COUNTS)
    bad_labels = LabeledData(X=data.X, y=np.array(["Otro"] * len(data.X), dtype=np.str_))
    with pytest.raises(ValueError, match="fuera de CLASSES"):
        split_data(bad_labels, calibration_size=0.3, test_size=0.2, random_state=0)
    dup = LabeledData(X=data.X.set_axis([1] * len(data.X)), y=data.y)
    with pytest.raises(ValueError, match="único"):
        split_data(dup, calibration_size=0.3, test_size=0.2, random_state=0)


def test_class_counts_includes_zero_rows_for_absent_classes():
    train = make_data({"Normal": 5, "HDF": 2}, start_uid=1)
    calibration = make_data({"Normal": 3}, start_uid=100)
    test = make_data({"Normal": 2, "HDF": 1}, start_uid=200)
    table = class_counts(DataSplits(train=train, calibration=calibration, test=test))
    assert list(table.index) == list(CLASSES)
    assert list(table.columns) == ["train", "calibration", "test", "total"]
    assert table.loc["HDF"].tolist() == [2, 0, 1, 3]
    assert table.loc["TWF"].tolist() == [0, 0, 0, 0]
    assert table["total"].sum() == 13


@pytest.mark.parametrize(
    "n, alpha, expected",
    [(0, 0.05, 1), (19, 0.05, 19), (18, 0.05, 19), (9, 0.1, 9), (8, 0.1, 9), (4, 0.2, 4)],
)
def test_conformal_rank_known_values(n, alpha, expected):
    assert conformal_rank(n, alpha=alpha) == expected


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.2, 1.5])
def test_conformal_rank_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        conformal_rank(10, alpha=alpha)


def test_calibration_feasibility_flags_small_and_absent_classes():
    train = make_data({"Normal": 30, "TWF": 5, "HDF": 5, "PWF": 5, "OSF": 5}, start_uid=1)
    calibration = make_data({"Normal": 40, "TWF": 18, "HDF": 19, "PWF": 9}, start_uid=100)
    test = make_data({"Normal": 5, "TWF": 1, "HDF": 1, "PWF": 1, "OSF": 1}, start_uid=300)
    splits = DataSplits(train=train, calibration=calibration, test=test)

    table = calibration_feasibility(splits, alpha=0.05)
    assert list(table.index) == list(CLASSES)
    assert table.loc["TWF", "n_calibration"] == 18
    assert table.loc["TWF", "rank"] == 19
    assert not table.loc["TWF", "finite_threshold_possible"]
    assert table.loc["HDF", "rank"] == 19
    assert table.loc["HDF", "finite_threshold_possible"]
    assert table.loc["OSF", "n_calibration"] == 0
    assert table.loc["OSF", "rank"] == 1
    assert not table.loc["OSF", "finite_threshold_possible"]

    relaxed = calibration_feasibility(splits, alpha=0.1)
    assert relaxed.loc["PWF", "rank"] == 9
    assert relaxed.loc["PWF", "finite_threshold_possible"]
    assert relaxed.loc["TWF", "finite_threshold_possible"]
