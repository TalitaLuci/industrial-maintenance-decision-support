"""Cria critérios de demonstração e combina AHP + TOPSIS."""
from pathlib import Path
import numpy as np
import pandas as pd
from .ahp import ahp_weights, example_balanced_matrix
from .topsis import topsis

CRITERIA = [
    "urgency",
    "operational_criticality",
    "intervention_cost",
    "downtime_impact",
]


def build_demo_criteria(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói critérios hipotéticos, transparentes e reproduzíveis.
    Criticidade/custos são regras de demonstração baseadas em ID de motor,
    não características observadas da NASA ou de motores reais.
    """
    df = predictions.copy()
    # Menor RUL = maior urgência; transformação positiva com maior valor = maior prioridade.
    df["urgency"] = 1.0 / (1.0 + df["rul_pred"].clip(lower=0))
    # Cenários sintéticos determinísticos apenas para exercitar o método:
    # não inferem criticidade real do motor.
    ids = df["unit_nr"].astype(int)
    df["operational_criticality"] = 1 + (ids % 3)  # escala ilustrativa 1–3
    df["intervention_cost"] = 1 + ((ids * 7) % 10)  # índice de custo hipotético 1–10
    df["downtime_impact"] = 1 + ((ids * 3) % 10)   # índice de impacto hipotético 1–10
    return df


def run_ranking(predictions: pd.DataFrame, profile: str = "balanced") -> tuple[pd.DataFrame, dict]:
    criteria = build_demo_criteria(predictions)
    matrices = {
        "balanced": example_balanced_matrix(),
        "safety_first": np.array([
            [1, 4, 5, 4],
            [1/4, 1, 2, 2],
            [1/5, 1/2, 1, 1/2],
            [1/4, 1/2, 2, 1],
        ], dtype=float),
        "cost_first": np.array([
            [1, 1/2, 1/3, 1/2],
            [2, 1, 1/2, 1],
            [3, 2, 1, 2],
            [2, 1, 1/2, 1],
        ], dtype=float),
    }
    if profile not in matrices:
        raise ValueError(f"Perfil desconhecido: {profile}")
    ahp = ahp_weights(matrices[profile])
    decision_matrix = criteria[CRITERIA].astype(float)
    # Critérios benefit: urgência, criticidade e impacto de indisponibilidade.
    # Custo de intervenção é custo: menor índice recebe preferência.
    ranked = topsis(
        decision_matrix,
        ahp["weights"],
        benefit_criteria=["urgency", "operational_criticality", "downtime_impact"],
    )
    out = criteria.join(ranked)
    return out.sort_values("rank"), {
        "profile": profile,
        "criteria": CRITERIA,
        "weights": dict(zip(CRITERIA, ahp["weights"].tolist())),
        "consistency_ratio": ahp["consistency_ratio"],
        "consistency_acceptable_reference": ahp["consistency_acceptable_reference"],
        "note": "Criticidade, custo e impacto são índices hipotéticos determinísticos.",
    }


def save_ranking(predictions_path: str | Path, reports_dir: str | Path, profile="balanced"):
    predictions = pd.read_csv(predictions_path)
    ranked, metadata = run_ranking(predictions, profile=profile)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(reports_dir / f"ranking_{profile}.csv", index=False)
    pd.Series(metadata["weights"], name="weight").to_csv(
        reports_dir / f"weights_{profile}.csv", header=True
    )
    return ranked, metadata
