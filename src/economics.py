"""Cenários econômicos ILUSTRATIVOS: todos os valores são hipóteses, não dados reais.

Pergunta: intervir agora ou adiar a intervenção?
- Intervir agora descarta parte da vida útil restante (custo de vida útil desperdiçada).
- Adiar mantém essa vida útil, mas aceita a probabilidade p de falha antes da intervenção
  (custo adicional de falha não planejada).
Em ambos os casos há custo de intervenção e de parada planejada.
"""
import pandas as pd


def wasted_life_cost(rul_pred: float, replacement_value: float,
                     reference_life: float = 125.0) -> float:
    """Valor hipotético da vida útil descartada ao intervir agora.

    Proporcional à RUL prevista (limitada à vida de referência).
    """
    if rul_pred < 0 or replacement_value < 0 or reference_life <= 0:
        raise ValueError("Valores inválidos para o custo de vida útil desperdiçada.")
    return replacement_value * min(rul_pred, reference_life) / reference_life


def compare_cost_scenarios(
    intervention_cost: float,
    downtime_cost: float,
    failure_cost: float,
    probability_of_failure_if_delayed: float,
    wasted_life: float = 0.0,
) -> pd.DataFrame:
    """Custo esperado de intervir agora vs adiar, em unidades monetárias hipotéticas."""
    values = [intervention_cost, downtime_cost, failure_cost,
              probability_of_failure_if_delayed, wasted_life]
    if any(v < 0 for v in values):
        raise ValueError("Os valores devem ser não negativos.")
    if probability_of_failure_if_delayed > 1:
        raise ValueError("A probabilidade hipotética deve estar entre 0 e 1.")
    base = intervention_cost + downtime_cost
    return pd.DataFrame([
        {
            "scenario": "intervene_now",
            "intervention_cost": intervention_cost,
            "downtime_cost": downtime_cost,
            "wasted_life_cost": wasted_life,
            "expected_failure_cost": 0.0,
            "total_cost": base + wasted_life,
        },
        {
            "scenario": "delay_intervention",
            "intervention_cost": intervention_cost,
            "downtime_cost": downtime_cost,
            "wasted_life_cost": 0.0,
            "expected_failure_cost": probability_of_failure_if_delayed * failure_cost,
            "total_cost": base + probability_of_failure_if_delayed * failure_cost,
        },
    ])


def break_even_probability(wasted_life: float, failure_cost: float) -> float:
    """Probabilidade de falha em que intervir agora e adiar custam o mesmo.

    p* = custo de vida útil desperdiçada / custo da falha. Se a probabilidade de
    falhar antes da intervenção for menor que p*, adiar custa menos (nas premissas).
    """
    if wasted_life < 0 or failure_cost <= 0:
        raise ValueError("Custo de falha deve ser positivo e a vida desperdiçada não negativa.")
    return min(1.0, wasted_life / failure_cost)
