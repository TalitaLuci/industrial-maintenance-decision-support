"""Critérios de decisão (um observado, três hipotéticos) e priorização AHP + TOPSIS."""
from pathlib import Path
import numpy as np
import pandas as pd
from .ahp import ahp_weights
from .topsis import topsis

# Ordem dos critérios nas matrizes AHP.
CRITERIA = ["urgency", "operational_criticality", "intervention_cost", "failure_impact"]
# Urgência, criticidade e impacto operacional da falha: quanto maior, maior a prioridade (benefício).
# "Impacto operacional da falha" = gravidade esperada de uma falha NÃO planejada deste motor
# (hipotético). Não é o custo da parada planejada para manutenção, que aparece em
# src/economics.py como `downtime_cost` e vale para os dois cenários (agir agora ou adiar).
# Custo de intervenção: quanto menor, melhor (custo).
BENEFIT_CRITERIA = ["urgency", "operational_criticality", "failure_impact"]

URGENCY_HORIZON = 125  # ciclos; horizonte de planejamento hipotético
DEFAULT_URGENCY_METHOD = "linear"

# Julgamentos AHP (hipotéticos) de cada perfil de decisor.
# Linhas/colunas na ordem de CRITERIA; a[i,j] = importância de i sobre j.
PROFILE_MATRICES = {
    "balanced": np.ones((4, 4), dtype=float),
    "safety_first": np.array([
        [1,   3,   7,   4],
        [1/3, 1,   5,   2],
        [1/7, 1/5, 1,   1/3],
        [1/4, 1/2, 3,   1],
    ], dtype=float),
    "cost_first": np.array([
        [1,   1/2, 1/4, 1],
        [2,   1,   1/3, 2],
        [4,   3,   1,   4],
        [1,   1/2, 1/4, 1],
    ], dtype=float),
}


def urgency_from_rul(rul_pred, method: str = DEFAULT_URGENCY_METHOD,
                     horizon: float = URGENCY_HORIZON) -> pd.Series:
    """Transforma RUL prevista em urgência (maior = mais urgente).

    - "linear":  1 - RUL/horizonte, truncada em [0, 1]. Interpretação: fração do
                 horizonte de planejamento que já foi consumida.
    - "inverse": 1 / (1 + RUL). Concentra os valores perto de zero.
    """
    rul = pd.Series(rul_pred, dtype=float).clip(lower=0)
    if method == "linear":
        return (1.0 - rul / horizon).clip(lower=0.0, upper=1.0)
    if method == "inverse":
        return 1.0 / (1.0 + rul)
    raise ValueError(f"Método de urgência desconhecido: {method}")


def build_demo_criteria(predictions: pd.DataFrame,
                        urgency_method: str = DEFAULT_URGENCY_METHOD,
                        seed: int = 42) -> pd.DataFrame:
    """Monta a matriz de critérios.

    Observado (previsto pelo modelo): urgência, a partir da RUL prevista.
    Hipotéticos: criticidade (1-3), índice de custo (1-10) e impacto operacional da falha
    (1-10), sorteados de forma independente e reprodutível por motor (seed).
    Não descrevem motores reais.
    """
    df = predictions.copy()
    df["urgency"] = urgency_from_rul(df["rul_pred"].to_numpy(), urgency_method).to_numpy()
    unit_ids = np.sort(df["unit_nr"].astype(int).unique())
    rng = np.random.default_rng(seed)
    hypothetical = pd.DataFrame({
        "operational_criticality": rng.integers(1, 4, len(unit_ids)),
        "intervention_cost": rng.integers(1, 11, len(unit_ids)),
        "failure_impact": rng.integers(1, 11, len(unit_ids)),
    }, index=unit_ids)
    ids = df["unit_nr"].astype(int)
    for col in hypothetical.columns:
        df[col] = ids.map(hypothetical[col]).to_numpy()
    return df


def run_ranking(predictions: pd.DataFrame, profile: str = "balanced",
                urgency_method: str = DEFAULT_URGENCY_METHOD) -> tuple[pd.DataFrame, dict]:
    if profile not in PROFILE_MATRICES:
        raise ValueError(f"Perfil desconhecido: {profile}")
    criteria = build_demo_criteria(predictions, urgency_method=urgency_method)
    ahp = ahp_weights(PROFILE_MATRICES[profile])
    ranked = topsis(criteria[CRITERIA].astype(float), ahp["weights"],
                    benefit_criteria=BENEFIT_CRITERIA)
    out = criteria.join(ranked).sort_values("rank")
    return out, {
        "profile": profile,
        "urgency_method": urgency_method,
        "criteria": CRITERIA,
        "weights": dict(zip(CRITERIA, ahp["weights"].tolist())),
        "consistency_ratio": ahp["consistency_ratio"],
        "consistency_acceptable_reference": ahp["consistency_acceptable_reference"],
        "note": "Criticidade, custo e impacto são índices hipotéticos (sorteio com semente fixa).",
    }


def compare_profiles(predictions: pd.DataFrame, urgency_method: str = DEFAULT_URGENCY_METHOD,
                     k: int = 10) -> dict:
    """Pontuações TOPSIS por perfil, correlação de postos e sobreposição do top-k."""
    scores = {}
    for profile in PROFILE_MATRICES:
        ranked, _ = run_ranking(predictions, profile, urgency_method)
        scores[profile] = ranked.set_index("unit_nr")["topsis_score"]
    scores = pd.DataFrame(scores)
    top = {p: set(scores[p].nlargest(k).index) for p in scores}
    overlap = pd.DataFrame({a: {b: len(top[a] & top[b]) for b in scores} for a in scores})
    return {
        "scores": scores,
        "spearman": scores.corr(method="spearman"),
        "top_k_overlap": overlap,
        "top_k_in_all_profiles": sorted(set.intersection(*top.values())),
    }


def save_ranking(predictions_path: str | Path, reports_dir: str | Path, profile="balanced",
                 urgency_method: str = DEFAULT_URGENCY_METHOD):
    predictions = pd.read_csv(predictions_path)
    ranked, metadata = run_ranking(predictions, profile, urgency_method)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(reports_dir / f"ranking_{profile}.csv", index=False)
    pd.Series(metadata["weights"], name="weight").to_csv(
        reports_dir / f"weights_{profile}.csv", header=True
    )
    return ranked, metadata
