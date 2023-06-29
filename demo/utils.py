from math import log10, acos, cos, sin, pi, atan2, sqrt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html

from constants import ALL_LAND_USE_COLS, CHART_COLS, SLIDER_PRECISION, LAND_USE_COLS
from constants import C3, C4, PRIMARY, SECONDARY, FIELDS

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
    map_fig.update_layout(margin=dict(l=0, r=10, t=0, b=0), showlegend=False)
    map_fig.update_geos(projection_scale=zoom, projection_type="orthographic", showcountries=True)
    return map_fig

def create_check_options(values):
    options = []
    for i in range(len(values)):
        options.append(
            {"label": [html.I(className="bi bi-lock"), html.Span(values[i])],
             "value": values[i]})
    return options

def compute_percent_change(context, presc):
    diffs = presc[LAND_USE_COLS].reset_index(drop=True) - context[LAND_USE_COLS].reset_index(drop=True)
    percent_changed = diffs[diffs > 0].sum(axis=1)
    percent_changed = percent_changed / context[LAND_USE_COLS].sum(axis=1)

    return percent_changed[0]

def create_treemap(data=pd.DataFrame(), type_context=True):
    """
    :param data: Pandas series of land use data
    :param type_context: If the title should be context or prescribed
    :return: Treemap figure
    """
    title = "Context" if type_context else "Prescribed"
    
    tree_params = dict(
        branchvalues = "total",
        sort=False,
        textinfo = "label+value",
        hoverinfo = "label+value+percent parent+percent entry+percent root",
        root_color="lightgrey"
    )

    labels, parents, values = None, None, None

    if data.empty:
        labels = [title]
        parents = [""]
        values = [1]

    else:
        total = data[ALL_LAND_USE_COLS].sum()
        c3 = data[C3].sum()
        c4 = data[C4].sum()
        crops = c3 + c4
        primary = data[PRIMARY].sum()
        secondary = data[SECONDARY].sum()
        fields = data[FIELDS].sum()

        labels = [title, "Nonland",
                "Crops", "C3", "C4", "c3ann", "c3nfx", "c3per", "c4ann", "c4per", 
                "Primary Vegetation", "primf", "primn", 
                "Secondary Vegetation", "secdf", "secdn",
                "Fields", "urban", "pastr", "range"]
        parents = ["", title,
                title, "Crops", "Crops", "C3", "C3", "C3", "C4", "C4",
                title, "Primary Vegetation", "Primary Vegetation",
                title, "Secondary Vegetation", "Secondary Vegetation",
                title, "Fields", "Fields", "Fields"]

        values =  [total + data["nonland"], data["nonland"],
                    crops, c3, c4, data["c3ann"], data["c3nfx"], data["c3per"], data["c4ann"], data["c4per"],
                    primary, data["primf"], data["primn"],
                    secondary, data["secdf"], data["secdn"],
                    fields, data["urban"], data["pastr"], data["range"]]
        
    assert(len(labels) == len(parents))
    assert(len(parents) == len(values))

    fig = go.Figure(
        go.Treemap(
            labels = labels,
            parents = parents,
            values=values,
            **tree_params
        )
    )
    colors = px.colors.qualitative.Plotly
    fig.update_layout(
        treemapcolorway = [colors[1], colors[4], colors[2], colors[7], colors[0]],
        margin=dict(t=0, b=0, l=10, r=10)
    )
    return fig

    

   
    
        