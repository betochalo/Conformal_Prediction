import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

from conformal_fault_inference_with_abstention.contracts import FEATURE_COLUMNS
from conformal_fault_inference_with_abstention.model import build_model


def test_model_training_and_unknown_type_do_not_refit_preprocessing():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=FEATURE_COLUMNS[1:])
    X.insert(0, "Type", np.tile(["L", "M"], 30))
    y = np.where(X["Torque"] > 0, "TWF", "Normal")
    model = build_model(random_state=42)
    with pytest.raises(NotFittedError):
        model.predict_proba(X)
    with threadpool_limits(limits=1):
        model.fit(X, y)
        encoder = model.named_steps["preprocessing"].named_transformers_["type"]
        before = encoder.categories_[0].copy()
        probe = X.iloc[:3].copy()
        probe["Type"] = "unknown"
        # Extra metadata must never become a predictor.
        probe["Machine failure"] = 1
        probabilities = model.predict_proba(probe)
        dense = model.named_steps["preprocessing"].transform(probe)
        assert isinstance(dense, np.ndarray)
        np.testing.assert_array_equal(dense[:, :2], np.zeros((3, 2)))
        np.testing.assert_array_equal(encoder.categories_[0], before)
        np.testing.assert_array_equal(before, ["L", "M"])
        assert probabilities.shape == (3, 2)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(
            model.predict(probe), model.classes_[probabilities.argmax(axis=1)]
        )
