from math import log10
import pandas as pd

from constants import ALL_LAND_USE_COLS, CHART_COLS, SLIDER_PRECISION

def add_nonland(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a nonland column that is the difference between 1 and
    ALL_LAND_USE_COLS.
        - Since sum isn't exactly 1 we just set to 0 if we get a negative.
    :param df: DataFrame of all land usage.
    :return: DataFrame with nonland column.
    """
    data = df[ALL_LAND_USE_COLS]
    nonland = 1 - data.sum(axis=1)
    nonland[nonland < 0] = 0
    assert((nonland >= 0).all())
    data['nonland'] = nonland
    return data[CHART_COLS]

def round_list(vals: list) -> list:
    """
    Rounds all the values of a list to the number of decimals that gets it
    to within SLIDER_PRECISION.
    :param vals: List of values to round.
    :return: List of rounded values.
    """
    decimals = int(-1 * log10(SLIDER_PRECISION))
    rounded = [round(val, decimals) for val in vals]
    return rounded