import pandas as pd
from src.data_processing import add_train_rul


def test_add_train_rul():
    df = pd.DataFrame({"unit_nr": [1, 1, 1, 2, 2], "time_cycles": [1, 2, 3, 1, 2]})
    out = add_train_rul(df)
    assert out["RUL"].tolist() == [2, 1, 0, 1, 0]


def test_rul_cap():
    df = pd.DataFrame({"unit_nr": [1, 1, 1], "time_cycles": [1, 2, 3]})
    out = add_train_rul(df, cap=1)
    assert out["RUL"].tolist() == [1, 1, 0]
