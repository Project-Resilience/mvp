from math import log10, acos, cos, sin, pi, atan2, sqrt
import pandas as pd
import plotly.express as px
from dash import html

from constants import ALL_LAND_USE_COLS, CHART_COLS, SLIDER_PRECISION, EARTH_RADIUS_KM

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

def create_map(df, lat_center, lon_center, zoom=10, color_idx = None):
    color = ["blue" for _ in range(len(df))]
    if color_idx:
        color[color_idx] = "red"
    color_seq = [px.colors.qualitative.Plotly[0], px.colors.qualitative.Plotly[1]]
    # TODO: Is this modification going to break things?
    df["color"] = color
    map_fig = px.scatter_geo(
        df, 
        lat="lat", 
        lon="lon", 
        color="color", 
        color_discrete_sequence=color_seq, 
        hover_data={"lat": True, "lon": True, "color": False},
        center={"lat": lat_center, "lon": lon_center}, 
        size_max=10
    )
    map_fig.update_layout(margin=dict(l=0, r=10, t=0, b=0), geo=dict(projection_scale=zoom), showlegend=False)
    return map_fig

def create_check_options(values):
    options = []
    for i in range(len(values)):
        options.append(
            {"label": [html.I(className="bi bi-lock"), html.Span(values[i])],
             "value": values[i]})
    return options

# From http://www.movable-type.co.uk/scripts/latlong.html
def latlon_dist(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371
    φ1 = lat1 * pi/180 # φ, λ in radians
    φ2 = lat2 * pi/180
    Δφ = (lat2-lat1) * pi/180
    Δλ = (lon2-lon1) * pi/180

    a = sin(Δφ/2) * sin(Δφ/2) + \
            cos(φ1) * cos(φ2) * \
            sin(Δλ/2) * sin(Δλ/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    d = R * c

    return d

def approx_area(coord):
    lat, lon = coord
    topleft = (lat+0.125, lon-0.125)
    topright = (lat+0.125, lon+0.125)
    botleft = (lat-0.125, lon-0.125)

    height = latlon_dist(topleft, botleft)
    width = latlon_dist(topleft, topright)
    # 100 to convert to hectares
    return height * width * 100
    

   
    
        