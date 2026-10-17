import pytest
from src.economics import (
    break_even_probability, compare_cost_scenarios, wasted_life_cost,
)


def test_both_scenarios_include_intervention_and_downtime():
    out = compare_cost_scenarios(100, 50, 1000, 0.2, wasted_life=80).set_index("scenario")
    assert out.loc["intervene_now", "total_cost"] == 100 + 50 + 80
    assert out.loc["delay_intervention", "total_cost"] == pytest.approx(100 + 50 + 0.2 * 1000)
    # A parada planejada não pode sumir do cenário "adiar" (viés a favor de adiar).
    assert out["downtime_cost"].eq(50).all()


def test_break_even_is_where_scenarios_cost_the_same():
    wasted, failure = 200.0, 2000.0
    p_star = break_even_probability(wasted, failure)
    out = compare_cost_scenarios(100, 50, failure, p_star, wasted).set_index("scenario")
    assert out.loc["intervene_now", "total_cost"] == pytest.approx(
        out.loc["delay_intervention", "total_cost"]
    )


def test_wasted_life_is_proportional_and_capped():
    assert wasted_life_cost(0, 400, 125) == 0
    assert wasted_life_cost(62.5, 400, 125) == pytest.approx(200)
    assert wasted_life_cost(500, 400, 125) == pytest.approx(400)  # limitada à vida de referência


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        compare_cost_scenarios(100, 50, 1000, 1.5)
    with pytest.raises(ValueError):
        compare_cost_scenarios(-1, 50, 1000, 0.2)
    with pytest.raises(ValueError):
        break_even_probability(10, 0)
