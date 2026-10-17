import numpy as np
from src.ahp import ahp_weights, example_balanced_matrix


def test_ahp_weights_sum_to_one():
    result = ahp_weights(example_balanced_matrix())
    assert np.isclose(result["weights"].sum(), 1.0)


def test_two_by_two_consistency_is_zero():
    result = ahp_weights(np.array([[1, 3], [1/3, 1]], dtype=float))
    assert np.isclose(result["consistency_ratio"], 0.0)


def test_hand_computed_weights_and_consistency():
    matrix = np.array([[1, 3, 5], [1/3, 1, 2], [1/5, 1/2, 1]], dtype=float)
    result = ahp_weights(matrix)
    assert np.allclose(result["weights"], [0.648, 0.230, 0.122], atol=1e-3)
    assert result["consistency_ratio"] < 0.01


def test_inconsistent_matrix_exceeds_reference():
    # A > B muito, B > C muito, mas C > A: julgamentos circulares.
    matrix = np.array([[1, 9, 1/9], [1/9, 1, 9], [9, 1/9, 1]], dtype=float)
    result = ahp_weights(matrix)
    assert result["consistency_ratio"] > 0.10
    assert result["consistency_acceptable_reference"] is False
