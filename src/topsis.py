"""Implementação didática de TOPSIS com orientação explícita dos critérios."""
import numpy as np
import pandas as pd


def topsis(
    decision_matrix: pd.DataFrame,
    weights,
    benefit_criteria: list[str],
) -> pd.DataFrame:
    """
    Retorna distância às soluções ideais e coeficiente de proximidade.
    Maior closeness significa maior proximidade à solução ideal positiva.
    `benefit_criteria` lista critérios em que valores maiores são preferíveis;
    os demais são tratados como critérios de custo (valores menores preferíveis).
    """
    if decision_matrix.empty:
        raise ValueError("A matriz de decisão está vazia.")
    if decision_matrix.isna().any().any():
        raise ValueError("A matriz de decisão contém valores ausentes.")
    if not np.isfinite(decision_matrix.to_numpy(dtype=float)).all():
        raise ValueError("A matriz deve conter valores finitos.")
    if (decision_matrix.to_numpy(dtype=float) < 0).any():
        raise ValueError("Esta implementação espera critérios não negativos.")
    cols = list(decision_matrix.columns)
    w = np.asarray(weights, dtype=float)
    if len(w) != len(cols) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("Pesos devem ser não negativos, alinhados às colunas e somar 1.")

    matrix = decision_matrix.to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    weighted = normalized * w
    ideal_best, ideal_worst = np.empty(len(cols)), np.empty(len(cols))
    for j, col in enumerate(cols):
        if col in benefit_criteria:
            ideal_best[j] = weighted[:, j].max()
            ideal_worst[j] = weighted[:, j].min()
        else:
            ideal_best[j] = weighted[:, j].min()
            ideal_worst[j] = weighted[:, j].max()

    d_best = np.linalg.norm(weighted - ideal_best, axis=1)
    d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    denominator = d_best + d_worst
    closeness = np.divide(
        d_worst, denominator,
        out=np.full_like(d_worst, 0.5),
        where=denominator > 0,
    )
    result = pd.DataFrame({
        "distance_to_ideal_best": d_best,
        "distance_to_ideal_worst": d_worst,
        "topsis_score": closeness,
    }, index=decision_matrix.index)
    result["rank"] = result["topsis_score"].rank(method="min", ascending=False).astype(int)
    return result.sort_values(["rank", "topsis_score"], ascending=[True, False])
