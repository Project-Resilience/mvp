"""
Utility functions for the demo application.
"""
import pandas as pd

import app.constants as app_constants
from data import constants


def add_nonland(data: pd.Series) -> pd.Series:
    """
    Adds a nonland column that is the difference between 1 and
    LAND_USE_COLS.
    Note: Since sum isn't exactly 1 we just set to 0 if we get a negative.
    :param data: pd Series containing land use data.
    :return: pd Series with nonland column added.
    """
    data = data[constants.LAND_USE_COLS]
    nonland = 1 - data.sum() if data.sum() <= 1 else 0
    data['nonland'] = nonland
    return data[app_constants.CHART_COLS]
