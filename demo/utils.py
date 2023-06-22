from math import log10
import pandas as pd
import plotly.express as px
from dash import html

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

def create_map(df, lat_center, lon_center, zoom=10):
    map_fig = px.scatter_geo(df, lat="lat", lon="lon", center={"lat": lat_center, "lon": lon_center}, size_max=10)
    map_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), geo=dict(projection_scale=zoom))
    return map_fig

def create_check_options(values):
    options = []
    for i in range(len(values)):
        options.append(
            {"label": [html.I(className="bi bi-lock"), html.Span(values[i])],
             "value": values[i]})
    return options
    
        