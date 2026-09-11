import numpy as np
import pytest

from conformal_fault_inference_with_abstention.contracts import PredictionSets
from conformal_fault_inference_with_abstention.evaluation import evaluate_sets


def test_hand_calculated_metrics_with_missing_class():
    sets = PredictionSets(
        np.array(["B", "A", "C", "D"]),
        np.array(
            [
                [False, False, False, False],
                [False, True, False, False],
                [True, True, False, False],
                [True, True, True, False],
            ]
        ),
    )
    report = evaluate_sets(["A", "A", "B", "C"], sets)
    assert report.n_samples == 4
    assert report.marginal_coverage == 0.75
    assert report.mean_set_size == 1.5
    assert report.abstention_rate == 0.75
    assert report.assisted_rate == 0.25
    assert report.human_rate == 0.5
    table = report.by_class.set_index("class")
    assert table.loc["A", "support"] == 2
    assert table.loc["A", "covered"] == 1
    assert table.loc["A", "coverage"] == 0.5
    assert table.loc["A", "mean_set_size"] == 0.5
    assert table.loc["A", "abstention_rate"] == 0.5
    assert table.loc["B", "coverage"] == 1
    assert table.loc["D", "support"] == 0
    assert table.loc["D", "covered"] == 0
    assert table.loc["D", ["coverage", "mean_set_size", "abstention_rate"]].isna().all()


@pytest.mark.parametrize("labels", [[], ["unknown"], [["A"]]])
def test_invalid_evaluation_labels(labels):
    with pytest.raises(ValueError):
        evaluate_sets(labels, PredictionSets(np.array(["A"]), np.array([[True]])))


def test_empty_evaluation_rejected():
    with pytest.raises(ValueError):
        evaluate_sets([], PredictionSets(np.array(["A"]), np.empty((0, 1), dtype=bool)))


def test_automatic_decision_errors_by_hand():
    from conformal_fault_inference_with_abstention.evaluation import automatic_decision_errors

    sets = PredictionSets(
        np.array(["Normal", "TWF", "HDF"]),
        np.array(
            [
                [True, False, False],  # Normal correcto, automático
                [False, True, False],  # Normal → TWF, automático incorrecto
                [False, True, False],  # TWF correcto, automático
                [True, True, False],  # asistido, no cuenta
                [False, False, False],  # humano, no cuenta
                [False, False, True],  # TWF → HDF, automático incorrecto
            ]
        ),
    )
    y = ["Normal", "Normal", "TWF", "TWF", "HDF", "TWF"]
    table = automatic_decision_errors(y, sets).set_index("class")
    assert table.loc["Normal", "automatic"] == 2
    assert table.loc["Normal", "errors"] == 1
    assert table.loc["Normal", "error_rate"] == 0.5
    assert table.loc["Normal", "predicted_as"] == "TWF"
    assert table.loc["TWF", "automatic"] == 2
    assert table.loc["TWF", "errors"] == 1
    assert table.loc["TWF", "predicted_as"] == "HDF"
    assert table.loc["HDF", "automatic"] == 0
    assert np.isnan(table.loc["HDF", "error_rate"])
    assert table.loc["HDF", "predicted_as"] == ""

    report = evaluate_sets(y, sets)
    assert report.automatic_count == 4
    assert report.automatic_errors == 2
    assert report.automatic_error_rate == 0.5
    assert list(report.automatic_errors_by_class["class"]) == ["Normal", "TWF", "HDF"]


def test_automatic_error_rate_is_nan_without_automatic_decisions():
    sets = PredictionSets(np.array(["A", "B"]), np.array([[True, True], [False, False]]))
    report = evaluate_sets(["A", "B"], sets)
    assert report.automatic_count == 0
    assert np.isnan(report.automatic_error_rate)
