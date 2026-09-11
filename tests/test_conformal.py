import numpy as np
import pytest

from conformal_fault_inference_with_abstention.conformal import (
    MondrianConformal,
    SplitConformal,
    conformal_quantile,
    conformal_rank,
)


def test_quantile_is_corrected_order_statistic_without_mutation():
    scores = np.array([0.75, 0.125, 0.5, 0.25])
    original = scores.copy()
    assert conformal_quantile(scores, alpha=0.4) == 0.5  # rank=3, not interpolation
    np.testing.assert_array_equal(scores, original)
    assert conformal_quantile([0.25, 0.25, 0.5], alpha=0.5) == 0.25
    assert np.isinf(conformal_quantile(scores, alpha=0.05))
    assert np.isinf(conformal_quantile([], alpha=0.1))
    assert conformal_rank(19, alpha=0.05) == 19
    assert conformal_rank(14, alpha=0.05) == 15


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan, np.inf])
def test_invalid_alpha(alpha):
    for constructor in (SplitConformal, MondrianConformal):
        with pytest.raises(ValueError):
            constructor(alpha=alpha)
    with pytest.raises(ValueError):
        conformal_quantile([], alpha=alpha)


@pytest.mark.parametrize("scores", [[[0.1]], [np.nan], [np.inf]])
def test_invalid_scores(scores):
    with pytest.raises(ValueError):
        conformal_quantile(scores, alpha=0.1)


def test_split_manual_threshold_ties_and_empty_sets():
    predictor = SplitConformal(alpha=0.5)
    # True-label scores: .125, .25, .5. Corrected rank=2 => q=.25.
    predictor.calibrate(
        [[0.875, 0.125], [0.75, 0.25], [0.5, 0.5]], ["B", "B", "A"], classes=["B", "A"]
    )
    assert predictor.threshold_ == 0.25
    result = predictor.predict_set([[0.75, 0.25], [0.5, 0.5], [0.125, 0.875]], classes=["B", "A"])
    np.testing.assert_array_equal(result.mask, [[True, False], [False, False], [False, True]])
    np.testing.assert_array_equal(result.classes, ["B", "A"])


def test_mondrian_groups_true_labels_and_handles_absent_class():
    predictor = MondrianConformal(alpha=0.5)
    # A scores .125,.25 => q_A=.25; B scores .75,.5 => q_B=.75.
    # Both B calibration rows predict A, detecting grouping by predicted label.
    predictor.calibrate(
        [[0.875, 0.125, 0], [0.75, 0.25, 0], [0.75, 0.25, 0], [0.5, 0.5, 0]],
        ["A", "A", "B", "B"],
        classes=["A", "B", "C"],
    )
    np.testing.assert_array_equal(predictor.thresholds_, [0.25, 0.75, np.inf])
    result = predictor.predict_set([[0.75, 0.25, 0], [0.5, 0.125, 0.375]], classes=["A", "B", "C"])
    np.testing.assert_array_equal(result.mask, [[True, True, True], [False, False, True]])


@pytest.mark.parametrize("constructor", [SplitConformal, MondrianConformal])
def test_state_validation_and_recalibration(constructor):
    predictor = constructor(alpha=0.5)
    with pytest.raises(RuntimeError):
        predictor.predict_set([[1, 0]], classes=["A", "B"])
    classes = np.array(["A", "B"])
    assert predictor.calibrate([[1, 0], [0, 1]], ["A", "B"], classes=classes) is predictor
    classes[0] = "X"
    with pytest.raises(ValueError):
        predictor.predict_set([[0.5, 0.5]], classes=["B", "A"])
    with pytest.raises(ValueError):
        predictor.predict_set([[1]], classes=["A", "B"])
    with pytest.raises(ValueError):
        predictor.calibrate([[1, 0]], ["unknown"], classes=["A", "B"])
    assert not predictor.predict_set([[0.5, 0.5]], classes=["A", "B"]).mask.any()
    predictor.calibrate(np.empty((0, 2)), [], classes=["B", "A"])
    assert predictor.predict_set([[0.5, 0.5]], classes=["B", "A"]).mask.all()
    assert predictor.predict_set(np.empty((0, 2)), classes=["B", "A"]).mask.shape == (0, 2)


@pytest.mark.parametrize("constructor", [SplitConformal, MondrianConformal])
def test_prediction_class_array_does_not_expose_internal_state(constructor):
    predictor = constructor(alpha=0.5).calibrate([[1, 0]], ["A"], classes=["A", "B"])
    result = predictor.predict_set([[1, 0]], classes=["A", "B"])
    result.classes[0] = "X"
    assert predictor.classes_[0] == "A"
