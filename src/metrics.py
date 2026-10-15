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
