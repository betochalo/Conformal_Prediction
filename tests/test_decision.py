import numpy as np
import pytest

from conformal_fault_inference_with_abstention.contracts import PredictionSets
from conformal_fault_inference_with_abstention.decision import route_predictions


def test_all_routes_and_empty_batch():
    sets = PredictionSets(
        np.array(["A", "B", "C"]),
        np.array(
            [[False, False, False], [True, False, False], [True, True, False], [True, True, True]]
        ),
    )
    np.testing.assert_array_equal(
        route_predictions(sets), ["human", "automatic", "assisted", "human"]
    )
    assert route_predictions(PredictionSets(sets.classes, np.empty((0, 3), dtype=bool))).size == 0


@pytest.mark.parametrize(
    "classes, mask",
    [
        (["A"], [[1]]),
        (["A"], [True]),
        (["A"], [[True, False]]),
        (["A", "A"], [[True, False]]),
        ([], np.empty((1, 0), dtype=bool)),
    ],
)
def test_reject_invalid_sets(classes, mask):
    with pytest.raises(ValueError):
        route_predictions(PredictionSets(np.asarray(classes), np.asarray(mask)))
