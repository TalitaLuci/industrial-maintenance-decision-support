"""Engenharia de atributos retrospectivos, sem usar ciclos futuros."""
import pandas as pd
from .data_processing import ID_COL, CYCLE_COL, SETTING_COLS, SENSOR_COLS


def make_window_features(
    df: pd.DataFrame,
    window: int = 5,
    include_current: bool = True,
) -> pd.DataFrame:
    """
    Cria atributos por motor ordenado no tempo.
    Usa observações atuais/passadas apenas. A média e o desvio são calculados
    em janela retrospectiva; a tendência usa diferença entre valor atual e
    valor de `window` ciclos antes, quando disponível.
    """
    if window < 2:
        raise ValueError("window deve ser >= 2.")
    required = {ID_COL, CYCLE_COL, *SETTING_COLS, *SENSOR_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")

    base = df.sort_values([ID_COL, CYCLE_COL]).copy()
    parts = []
    feature_cols = [*SETTING_COLS, *SENSOR_COLS]
    for col in feature_cols:
        grouped = base.groupby(ID_COL, sort=False)[col]
        if include_current:
            base[f"{col}_current"] = base[col]
        base[f"{col}_mean_{window}"] = grouped.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        base[f"{col}_std_{window}"] = grouped.transform(
            lambda s: s.rolling(window, min_periods=2).std()
        ).fillna(0.0)
        base[f"{col}_trend_{window}"] = grouped.transform(
            lambda s: s - s.shift(window - 1)
        ).fillna(0.0)
    excluded = set(feature_cols)
    excluded.update({"RUL"})
    feature_names = [
        c for c in base.columns
        if c not in excluded and c not in {ID_COL}
        and (c == CYCLE_COL or c.endswith(f"_current")
             or f"_mean_{window}" in c or f"_std_{window}" in c
             or f"_trend_{window}" in c)
    ]
    # O número absoluto do ciclo pode carregar sinal temporal; mantido de forma explícita.
    feature_names = list(dict.fromkeys(feature_names))
    feature_names = [c for c in feature_names if c != CYCLE_COL]
    return base[[ID_COL, CYCLE_COL, *feature_names, *(["RUL"] if "RUL" in base else [])]].copy()


def get_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Seleciona apenas features numéricas; exclui identificador e alvo."""
    excluded = {ID_COL, "RUL"}
    return [
        c for c in feature_df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(feature_df[c])
    ]


def drop_constant_columns(train_feature_df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Remove colunas sem variação no TREINO (inclui derivadas de sensores constantes).

    A lista resultante é decidida só com o treino e reaproveitada no teste.
    """
    return [c for c in columns if train_feature_df[c].nunique(dropna=False) > 1]
