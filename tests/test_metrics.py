import numpy as np
from src.metrics import cmapss_asymmetric_score, regression_metrics


def test_score_is_zero_for_perfect_prediction():
    assert cmapss_asymmetric_score([10, 50], [10, 50]) == 0.0


def test_overestimation_is_penalized_more_than_underestimation():
    over = cmapss_asymmetric_score([50], [60])    # erro +10
    under = cmapss_asymmetric_score([50], [40])   # erro -10
    assert over > under
    assert np.isclose(over, np.exp(10 / 10) - 1)
    assert np.isclose(under, np.exp(10 / 13) - 1)


def test_regression_metrics_known_values():
    m = regression_metrics([0, 0], [3, 4])
    assert np.isclose(m["mae"], 3.5)
    assert np.isclose(m["rmse"], np.sqrt((9 + 16) / 2))
