import pandas as pd

from constants import ALL_LAND_USE_COLS, CHART_COLS

def add_nonland(df: pd.DataFrame) -> pd.DataFrame:
    data = df[ALL_LAND_USE_COLS]
    nonland = 1 - data.sum(axis=1)
    nonland[nonland < 0] = 0
    assert((nonland >= 0).all())
    data['nonland'] = nonland
    return data[CHART_COLS]