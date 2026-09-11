import numpy as np
import pytest

from conformal_fault_inference_with_abstention.scores import (
    inverse_probability,
    true_label_scores,
)


def test_scores_use_explicit_class_order():
    proba = np.array([[0.75, 0.25], [0.125, 0.875]])
    np.testing.assert_array_equal(inverse_probability(proba), [[0.25, 0.75], [0.875, 0.125]])
    np.testing.assert_array_equal(
        true_label_scores(proba, np.array(["B", "A"]), classes=np.array(["B", "A"])),
        [0.25, 0.125],
    )


@pytest.mark.parametrize(
    "proba", [[0.5, 0.5], [[0.2, 0.3]], [[-0.1, 1.1]], [[np.nan, 0]], [[np.inf, 0]], [[]]]
)
def test_invalid_probabilities(proba):
    with pytest.raises(ValueError):
        inverse_probability(proba)


@pytest.mark.parametrize(
    ("y", "classes"),
    [
        (["C"], ["A", "B"]),
        (["A"], ["A", "A"]),
        ([], ["A", "B"]),
        ([["A"]], ["A", "B"]),
        (["A"], ["A"]),
        (["A"], ["", "B"]),
    ],
)
def test_invalid_labels_and_classes(y, classes):
    with pytest.raises(ValueError):
        true_label_scores([[0.5, 0.5]], y, classes=classes)


def test_empty_scores_preserve_shapes():
    proba = np.empty((0, 2))
    assert inverse_probability(proba).shape == (0, 2)
    assert true_label_scores(proba, [], classes=["A", "B"]).shape == (0,)
