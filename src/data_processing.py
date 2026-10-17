"""Leitura e validação básica dos arquivos NASA C-MAPSS."""
from pathlib import Path
import pandas as pd

ID_COL = "unit_nr"
CYCLE_COL = "time_cycles"
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = [ID_COL, CYCLE_COL, *SETTING_COLS, *SENSOR_COLS]


def load_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Lê arquivo C-MAPSS sem cabeçalho, delimitado por espaços."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. Consulte o README para obter os dados."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Alguns espelhos dos arquivos incluem colunas vazias no final; descarta-as.
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Esperadas {len(COLUMNS)} colunas, mas foram encontradas {df.shape[1]} em {path}."
        )
    df.columns = COLUMNS
    df[ID_COL] = df[ID_COL].astype(int)
    df[CYCLE_COL] = df[CYCLE_COL].astype(int)
    return df


def add_train_rul(train: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Adiciona RUL por linha. cap=None mantém o alvo sem truncamento."""
    out = train.copy()
    max_cycle = out.groupby(ID_COL)[CYCLE_COL].transform("max")
    out["RUL"] = max_cycle - out[CYCLE_COL]
    if cap is not None:
        if cap <= 0:
            raise ValueError("cap deve ser positivo ou None.")
        out["RUL"] = out["RUL"].clip(upper=cap)
    return out


def audit_data(df: pd.DataFrame) -> dict:
    """Resumo de qualidade útil para inspeção inicial."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "engine_count": int(df[ID_COL].nunique()),
        "cycles_per_engine": df.groupby(ID_COL)[CYCLE_COL].agg(
            ["min", "max", "count"]
        ).describe().to_dict(),
        "constant_columns": [
            c for c in df.columns if df[c].nunique(dropna=False) <= 1
        ],
    }


def find_constant_columns(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    """Colunas sem nenhuma variação (um único valor) no DataFrame informado."""
    cols = columns if columns is not None else list(df.columns)
    return [c for c in cols if df[c].nunique(dropna=False) <= 1]
