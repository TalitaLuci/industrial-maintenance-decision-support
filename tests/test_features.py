import numpy as np
import pandas as pd
from src.data_processing import ID_COL, CYCLE_COL, SETTING_COLS, SENSOR_COLS
from src.features import make_window_features, get_feature_columns


def _toy(n_cycles=8, units=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in units:
        for c in range(1, n_cycles + 1):
            rows.append([u, c, *rng.normal(size=len(SETTING_COLS) + len(SENSOR_COLS))])
    return pd.DataFrame(rows, columns=[ID_COL, CYCLE_COL, *SETTING_COLS, *SENSOR_COLS])


def test_no_duplicate_columns():
    # Regressão: time_cycles aparecia duas vezes e quebrava sort_values/groupby.
    feats = make_window_features(_toy())
    assert feats.columns.is_unique
    assert get_feature_columns(feats).count(CYCLE_COL) == 1


def test_window_features_do_not_use_future_cycles():
    df = _toy()
    full = make_window_features(df).set_index([ID_COL, CYCLE_COL])
    # Altera apenas os ciclos finais: features dos ciclos anteriores não podem mudar.
    changed = df.copy()
    changed.loc[changed[CYCLE_COL] >= 7, SENSOR_COLS] += 100.0
    alt = make_window_features(changed).set_index([ID_COL, CYCLE_COL])
    early = full.index.get_level_values(CYCLE_COL) <= 6
    pd.testing.assert_frame_equal(full[early], alt[early])


def test_features_are_computed_per_engine():
    df = _toy()
    feats = make_window_features(df, window=3)
    first_rows = feats[feats[CYCLE_COL] == 1]
    # No primeiro ciclo de cada motor, a média móvel é o próprio valor atual.
    assert np.allclose(first_rows["s_2_mean_3"], first_rows["s_2_current"])
