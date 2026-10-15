import numpy as np
from src.ahp import ahp_weights, example_balanced_matrix


def test_ahp_weights_sum_to_one():
    result = ahp_weights(example_balanced_matrix())
    assert np.isclose(result["weights"].sum(), 1.0)


def test_two_by_two_consistency_is_zero():
    result = ahp_weights(np.array([[1, 3], [1/3, 1]], dtype=float))
    assert np.isclose(result["consistency_ratio"], 0.0)
