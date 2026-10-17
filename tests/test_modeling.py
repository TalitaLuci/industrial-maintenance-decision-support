import numpy as np
import pandas as pd
import pytest
from src.metrics import error_summary
from src.modeling import make_target


def test_make_target_caps_only_when_requested():
    rul = np.array([10, 125, 300])
    assert make_target(rul, None).tolist() == [10, 125, 300]
    assert make_target(rul, 125).tolist() == [10, 125, 125]


def test_error_summary_compares_with_real_rul_by_band():
    y_true = np.array([10, 60, 120, 300])
    y_pred = np.array([20, 50, 130, 125])  # erro: +10, -10, +10, -175
    out = error_summary(y_true, y_pred)
    assert out["mae_0_50"] == pytest.approx(10)
    assert out["mae_50_100"] == pytest.approx(10)
    assert out["mae_100_150"] == pytest.approx(10)
    # A linha de RUL 300 fica fora das métricas "<= 150":
    assert out["rmse_le150"] == pytest.approx(10)
    assert out["bias_le150"] == pytest.approx(10 / 3)
    assert out["mae"] == pytest.approx((10 + 10 + 10 + 175) / 4)
