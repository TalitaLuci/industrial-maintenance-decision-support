import numpy as np
import pandas as pd
import pytest
from src.ahp import ahp_weights
from src.decision import (
    CRITERIA, PROFILE_MATRICES, build_demo_criteria, compare_profiles,
    run_ranking, urgency_from_rul,
)


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "unit_nr": np.arange(1, 31),
        "last_observed_cycle": 100,
        "rul_pred": rng.uniform(5, 125, 30),
    })


@pytest.mark.parametrize("profile", list(PROFILE_MATRICES))
def test_profiles_are_consistent_and_weights_sum_to_one(profile):
    result = ahp_weights(PROFILE_MATRICES[profile])
    assert result["weights"].sum() == pytest.approx(1.0)
    assert result["consistency_ratio"] < 0.10


@pytest.mark.parametrize("method", ["linear", "inverse"])
def test_urgency_decreases_with_rul(method):
    urgency = urgency_from_rul([10, 50, 100], method)
    assert urgency.is_monotonic_decreasing


def test_linear_urgency_is_bounded():
    urgency = urgency_from_rul([-5, 0, 62.5, 125, 300], "linear", horizon=125)
    assert urgency.tolist() == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0])


def test_hypothetical_criteria_are_reproducible_and_independent_of_row_order(predictions):
    a = build_demo_criteria(predictions).set_index("unit_nr")
    b = build_demo_criteria(predictions.sample(frac=1, random_state=1)).set_index("unit_nr").loc[a.index]
    pd.testing.assert_frame_equal(
        a[["operational_criticality", "intervention_cost", "downtime_impact"]],
        b[["operational_criticality", "intervention_cost", "downtime_impact"]],
    )


def test_ranking_is_a_permutation_and_has_no_ties(predictions):
    ranked, meta = run_ranking(predictions, "safety_first")
    assert sorted(ranked["rank"]) == list(range(1, len(predictions) + 1))
    assert set(meta["weights"]) == set(CRITERIA)


def test_lowest_rul_engine_leads_safety_profile(predictions):
    ranked, _ = run_ranking(predictions, "safety_first")
    assert ranked.iloc[0]["rul_pred"] <= predictions["rul_pred"].quantile(0.15)


def test_compare_profiles_outputs_expected_shapes(predictions):
    out = compare_profiles(predictions, k=5)
    assert out["spearman"].shape == (3, 3)
    assert (np.diag(out["top_k_overlap"]) == 5).all()
