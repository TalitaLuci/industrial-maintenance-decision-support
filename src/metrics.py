"""Métricas de avaliação de RUL."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def cmapss_asymmetric_score(y_true, y_pred) -> float:
    """
    Escore assimétrico C-MAPSS. Menor é melhor.
    Convenção: erro = previsto - real. Superestimar RUL (erro > 0)
    recebe penalização mais severa.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    exponents = np.where(error < 0, -error / 13.0, error / 10.0)
    # Proteção numérica para valores extremos.
    exponents = np.clip(exponents, -50, 50)
    return float(np.sum(np.exp(exponents) - 1.0))


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "cmapss_score": cmapss_asymmetric_score(y_true, y_pred),
        "n": int(len(y_true)),
    }


RUL_BANDS = [(0, 50), (50, 100), (100, 150)]


def error_summary(y_true, y_pred, horizon: float = 150.0) -> dict:
    """Resumo por faixa de RUL real (erro = previsto - real).

    As métricas por faixa e "_le{horizon}" evitam que linhas de RUL muito alta,
    ausentes do teste oficial, dominem a média geral.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    out = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
    }
    within = y_true <= horizon
    out["rmse_le150"] = float(np.sqrt((err[within] ** 2).mean()))
    out["bias_le150"] = float(err[within].mean())
    for lo, hi in RUL_BANDS:
        mask = (y_true >= lo) & (y_true < hi)
        out[f"mae_{lo}_{hi}"] = float(np.abs(err[mask]).mean()) if mask.any() else float("nan")
    return out
