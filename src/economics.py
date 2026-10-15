"""Comparação de cenários econômicos hipotéticos; não são custos observados."""
import pandas as pd


def compare_cost_scenarios(
    intervention_cost: float,
    downtime_cost: float,
    failure_cost: float,
    probability_of_failure_if_delayed: float,
) -> pd.DataFrame:
    """
    Exemplo simplificado para explicitar hipóteses.
    Probabilidade precisa ser definida pelo usuário como hipótese de cenário;
    não é estimada nem calibrada pelo projeto.
    """
    values = [intervention_cost, downtime_cost, failure_cost, probability_of_failure_if_delayed]
    if any(v < 0 for v in values):
        raise ValueError("Os valores devem ser não negativos.")
    if probability_of_failure_if_delayed > 1:
        raise ValueError("A probabilidade hipotética deve estar entre 0 e 1.")
    return pd.DataFrame([
        {
            "scenario": "intervene_now",
            "assumed_intervention_cost": intervention_cost,
            "assumed_downtime_cost": downtime_cost,
            "assumed_expected_failure_cost": 0.0,
            "total_scenario_cost": intervention_cost + downtime_cost,
        },
        {
            "scenario": "delay_intervention",
            "assumed_intervention_cost": intervention_cost,
            "assumed_downtime_cost": 0.0,
            "assumed_expected_failure_cost": probability_of_failure_if_delayed * failure_cost,
            "total_scenario_cost": intervention_cost + probability_of_failure_if_delayed * failure_cost,
        },
    ])
