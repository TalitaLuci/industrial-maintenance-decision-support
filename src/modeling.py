"""Treinamento de baseline e modelo de RUL com validação agrupada por motor."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyRegressor

from .data_processing import load_cmapss_file, add_train_rul, ID_COL, CYCLE_COL
from .features import make_window_features, get_feature_columns
from .metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
WINDOW = 5


def prepare():
    train = load_cmapss_file(RAW / "train_FD001.txt")
    test = load_cmapss_file(RAW / "test_FD001.txt")
    rul_path = RAW / "RUL_FD001.txt"
    if not rul_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {rul_path}")
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    train = add_train_rul(train)
    train_feat = make_window_features(train, window=WINDOW)
    test_feat = make_window_features(test, window=WINDOW)
    return train_feat, test_feat, true_rul


def evaluate_group_cv(train_feat, feature_cols):
    """CV de desenvolvimento. Os motores são grupos; teste oficial não participa."""
    groups = train_feat[ID_COL].to_numpy()
    X = train_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_feat["RUL"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    rows = []
    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        # Baseline constante: mediana do RUL da partição de treino.
        baseline = DummyRegressor(strategy="median")
        baseline.fit(X.iloc[tr_idx], y[tr_idx])
        baseline_pred = baseline.predict(X.iloc[val_idx])
        model = HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        )
        model.fit(X.iloc[tr_idx], y[tr_idx])
        pred = np.maximum(0.0, model.predict(X.iloc[val_idx]))
        base_metrics = regression_metrics(y[val_idx], baseline_pred)
        model_metrics = regression_metrics(y[val_idx], pred)
        rows.append({
            "fold": fold,
            "baseline_mae": base_metrics["mae"],
            "baseline_rmse": base_metrics["rmse"],
            "baseline_cmapss_score": base_metrics["cmapss_score"],
            "model_mae": model_metrics["mae"],
            "model_rmse": model_metrics["rmse"],
            "model_cmapss_score": model_metrics["cmapss_score"],
            "validation_engines": int(train_feat.iloc[val_idx][ID_COL].nunique()),
        })
    return pd.DataFrame(rows)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    train_feat, test_feat, true_rul = prepare()
    feature_cols = get_feature_columns(train_feat)
    cv = evaluate_group_cv(train_feat, feature_cols)
    cv.to_csv(REPORTS / "cv_metrics.csv", index=False)

    # Depois da avaliação de desenvolvimento, ajusta um único modelo em todo o treino.
    X_train = train_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_feat["RUL"].to_numpy()
    model = HistGradientBoostingRegressor(
        max_iter=180, learning_rate=0.08, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42
    )
    model.fit(X_train, y_train)

    # Diagnóstico: verificar se existem colunas duplicadas.
    print("Colunas duplicadas:")
    print(test_feat.columns[test_feat.columns.duplicated()].tolist())

    print("\nTodas as colunas relacionadas a ciclos:")
    print([col for col in test_feat.columns if "cycle" in col.lower()])

    # Avaliação final: última observação disponível de cada motor de teste.
    last_rows = (
        test_feat.sort_values([ID_COL, CYCLE_COL])
        .groupby(ID_COL, as_index=False)
        .tail(1)
        .sort_values(ID_COL)
        .reset_index(drop=True)
    )

    if len(last_rows) != len(true_rul):
        raise ValueError(
            f"Há {len(last_rows)} motores de teste e {len(true_rul)} valores de RUL. "
            "Verifique a ordem/versão dos arquivos."
        )
    X_test = last_rows[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pred = np.maximum(0.0, model.predict(X_test))
    test_metrics = regression_metrics(true_rul, pred)

    predictions = pd.DataFrame({
        "unit_nr": last_rows[ID_COL].to_numpy(),
        "last_observed_cycle": last_rows[CYCLE_COL].to_numpy(),
        "rul_true": true_rul,
        "rul_pred": pred,
        "error_pred_minus_true": pred - true_rul,
    })
    predictions.to_csv(REPORTS / "test_predictions.csv", index=False)
    with open(REPORTS / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print("CV salvo em reports/cv_metrics.csv")
    print("Métricas do teste oficial:", json.dumps(test_metrics, indent=2))
    print("Previsões salvas em reports/test_predictions.csv")
    print("Não ajuste o modelo com base nos resultados do teste oficial.")


if __name__ == "__main__":
    main()
