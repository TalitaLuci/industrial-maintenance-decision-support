"""Cálculo de pesos AHP por autovetor principal."""
import numpy as np

# Índices de consistência aleatória de Saaty.
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp_weights(pairwise: np.ndarray) -> dict:
    matrix = np.asarray(pairwise, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A matriz AHP deve ser quadrada.")
    if matrix.shape[0] < 2:
        raise ValueError("A matriz deve conter pelo menos dois critérios.")
    if np.any(matrix <= 0):
        raise ValueError("As comparações devem ser positivas.")
    if not np.allclose(np.diag(matrix), 1.0):
        raise ValueError("A diagonal principal deve conter 1.")
    if not np.allclose(matrix * matrix.T, 1.0, atol=1e-6):
        raise ValueError("A matriz deve ser recíproca: a[i,j] = 1/a[j,i].")

    values, vectors = np.linalg.eig(matrix)
    idx = int(np.argmax(values.real))
    principal_value = float(values[idx].real)
    weights = np.abs(vectors[:, idx].real)
    weights = weights / weights.sum()
    n = matrix.shape[0]
    ci = 0.0 if n <= 2 else max(0.0, (principal_value - n) / (n - 1))
    ri = RI_TABLE.get(n, 1.49)
    cr = 0.0 if ri == 0 else ci / ri
    return {
        "weights": weights,
        "lambda_max": principal_value,
        "consistency_index": float(ci),
        "consistency_ratio": float(cr),
        "consistency_acceptable_reference": bool(cr <= 0.10),
    }


def example_balanced_matrix() -> np.ndarray:
    """Perfil equilibrado: os quatro critérios têm a mesma importância."""
    return np.ones((4, 4), dtype=float)
