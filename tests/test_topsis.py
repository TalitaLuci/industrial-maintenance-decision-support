import numpy as np
import pandas as pd
from src.topsis import topsis


def test_benefit_criterion_prefers_larger_value():
    matrix = pd.DataFrame({"benefit": [1.0, 10.0], "cost": [5.0, 5.0]}, index=["A", "B"])
    result = topsis(matrix, [0.8, 0.2], benefit_criteria=["benefit"])
    assert result.loc["B", "rank"] == 1


def test_cost_criterion_prefers_smaller_value():
    matrix = pd.DataFrame({"benefit": [5.0, 5.0], "cost": [1.0, 10.0]}, index=["A", "B"])
    result = topsis(matrix, [0.2, 0.8], benefit_criteria=["benefit"])
    assert result.loc["A", "rank"] == 1


def test_identical_alternatives_tie():
    matrix = pd.DataFrame({"x": [2.0, 2.0], "y": [4.0, 4.0]}, index=["A", "B"])
    result = topsis(matrix, [0.5, 0.5], benefit_criteria=["x", "y"])
    assert np.allclose(result["topsis_score"], [0.5, 0.5])


def test_hand_computed_case_and_orientation_inversion():
    # Caso pequeno calculado manualmente (veja o README/notebook 03).
    matrix = pd.DataFrame(
        {"urgency": [3.0, 5.0, 4.0], "cost": [200.0, 300.0, 100.0]}, index=["A", "B", "C"]
    )
    ok = topsis(matrix, [0.6, 0.4], benefit_criteria=["urgency"])
    assert np.allclose(
        ok.loc[["A", "B", "C"], "topsis_score"], [0.3477, 0.4425, 0.7305], atol=1e-3
    )
    assert ok.sort_values("rank").index.tolist() == ["C", "B", "A"]
    # Tratar o custo como benefício por engano deve inverter o resultado.
    wrong = topsis(matrix, [0.6, 0.4], benefit_criteria=["urgency", "cost"])
    assert wrong.sort_values("rank").index.tolist() == ["B", "A", "C"]
