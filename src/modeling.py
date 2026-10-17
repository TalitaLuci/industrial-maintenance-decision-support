"""Treinamento e avaliação de RUL com validação agrupada por motor.

Fluxo: validação cruzada por motor (desenvolvimento) -> escolha da variante ->
avaliação única no teste oficial. O teste nunca participa da escolha do modelo.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .data_processing import load_cmapss_file, add_train_rul, ID_COL, CYCLE_COL
from .features import make_window_features, get_feature_columns, drop_constant_columns
from .metrics import regression_metrics, error_summary

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
WINDOW = 5
# Variante escolhida no notebook 01 (validação cruzada). None = sem teto.
RUL_CAP = 125
HGB_PARAMS = dict(
    max_iter=180, learning_rate=0.08, max_leaf_nodes=15,
    l2_regularization=1.0, early_stopping=False, random_state=42,
)


def prepare():
    """Lê os dados e cria os atributos. A coluna RUL do treino é a RUL REAL (sem teto)."""
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


def select_features(train_feat: pd.DataFrame) -> list[str]:
    """Atributos numéricos do treino, sem colunas constantes."""
    return drop_constant_columns(train_feat, get_feature_columns(train_feat))


def make_target(rul_true, cap):
    """Alvo de treino: RUL real, opcionalmente limitada a `cap` ciclos."""
    rul_true = np.asarray(rul_true, dtype=float)
    return rul_true if cap is None else np.minimum(rul_true, cap)


def _clean(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def evaluate_group_cv(train_feat, feature_cols, cap=RUL_CAP, n_splits=5) -> pd.DataFrame:
    """CV agrupada por motor. As métricas comparam sempre com a RUL REAL (sem teto)."""
    X = _clean(train_feat[feature_cols])
    y_true = train_feat["RUL"].to_numpy(dtype=float)
    y_target = make_target(y_true, cap)
    groups = train_feat[ID_COL].to_numpy()
    rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y_true, groups), 1):
        baseline = DummyRegressor(strategy="median").fit(X.iloc[tr], y_target[tr])
        model = HistGradientBoostingRegressor(**HGB_PARAMS).fit(X.iloc[tr], y_target[tr])
        pred_base = baseline.predict(X.iloc[va])
        pred_model = np.maximum(0.0, model.predict(X.iloc[va]))
        row = {"fold": fold, "validation_engines": int(train_feat.iloc[va][ID_COL].nunique())}
        for prefix, pred in (("baseline", pred_base), ("model", pred_model)):
            for key, value in error_summary(y_true[va], pred).items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def fit_final(train_feat, feature_cols, cap=RUL_CAP):
    X = _clean(train_feat[feature_cols])
    y = make_target(train_feat["RUL"].to_numpy(dtype=float), cap)
    return HistGradientBoostingRegressor(**HGB_PARAMS).fit(X, y)


def last_observed_rows(test_feat: pd.DataFrame) -> pd.DataFrame:
    """Última observação disponível de cada motor de teste (ordenado por motor)."""
    return (
        test_feat.sort_values([ID_COL, CYCLE_COL])
        .groupby(ID_COL, as_index=False).tail(1)
        .sort_values(ID_COL).reset_index(drop=True)
    )


def predict_official_test(model, test_feat, true_rul, feature_cols):
    """Avaliação final no teste oficial (usar UMA vez, após escolher a variante)."""
    last_rows = last_observed_rows(test_feat)
    if len(last_rows) != len(true_rul):
        raise ValueError(
            f"Há {len(last_rows)} motores de teste e {len(true_rul)} valores de RUL. "
            "Verifique a ordem/versão dos arquivos."
        )
    pred = np.maximum(0.0, model.predict(_clean(last_rows[feature_cols])))
    predictions = pd.DataFrame({
        "unit_nr": last_rows[ID_COL].to_numpy(),
        "last_observed_cycle": last_rows[CYCLE_COL].to_numpy(),
        "rul_true": true_rul,
        "rul_pred": pred,
        "error_pred_minus_true": pred - true_rul,
    })
    metrics = regression_metrics(true_rul, pred)
    metrics["bias"] = float((pred - true_rul).mean())
    metrics["share_overestimated"] = float((pred > true_rul).mean())
    return predictions, metrics


def baseline_test_metrics(train_feat, true_rul, cap=RUL_CAP):
    """Baseline constante (mediana do alvo de treino) aplicado ao teste oficial."""
    median = float(np.median(make_target(train_feat["RUL"], cap)))
    return regression_metrics(true_rul, np.full(len(true_rul), median))


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    train_feat, test_feat, true_rul = prepare()
    feature_cols = select_features(train_feat)
    cv = evaluate_group_cv(train_feat, feature_cols, cap=RUL_CAP)
    cv.to_csv(REPORTS / "cv_metrics.csv", index=False)

    model = fit_final(train_feat, feature_cols, cap=RUL_CAP)
    predictions, metrics = predict_official_test(model, test_feat, true_rul, feature_cols)
    metrics["baseline"] = baseline_test_metrics(train_feat, true_rul)
    metrics["rul_cap"] = RUL_CAP
    predictions.to_csv(REPORTS / "test_predictions.csv", index=False)
    (REPORTS / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"CV (teto={RUL_CAP}) salvo em reports/cv_metrics.csv")
    print("Métricas do teste oficial:", json.dumps(metrics, indent=2))
    print("Previsões salvas em reports/test_predictions.csv")
    print("Não ajuste o modelo com base nos resultados do teste oficial.")


if __name__ == "__main__":
    main()
