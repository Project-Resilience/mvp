import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html

from constants import RECO_COLS
from constants import CHART_COLS
from constants import LAND_USE_COLS
from constants import C3
from constants import C4
from constants import PRIMARY
from constants import SECONDARY
from constants import FIELDS

def add_nonland(data: pd.Series) -> pd.Series:
    """
    Adds a nonland column that is the difference between 1 and
    LAND_USE_COLS.
    Note: Since sum isn't exactly 1 we just set to 0 if we get a negative.
    :param data: pd Series containing land use data.
    :return: pd Series with nonland column added.
    """
    data = data[LAND_USE_COLS]
    nonland = 1 - data.sum() if data.sum() <= 1 else 0
    data['nonland'] = nonland
    return data[CHART_COLS]


def create_map(df: pd.DataFrame, lat_center: float, lon_center: float, zoom=10, color_idx = None) -> go.Figure:
    """
    Creates map figure with data centered and zoomed in with appropriate point marked.
    :param df: DataFrame of data to plot. This dataframe has its index reset.
    :param lat_center: Latitude to center map on.
    :param lon_center: Longitude to center map on.
    :param zoom: Zoom level of map.
    :param color_idx: Index of point to color red in reset index.
    :return: Plotly figure
    """
    color_seq = [px.colors.qualitative.Plotly[0], px.colors.qualitative.Plotly[1]]

    # Add color column    
    color = ["blue" for _ in range(len(df))]
    if color_idx:
        color[color_idx] = "red"
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
    map_fig.update_layout(margin={"l": 0, "r": 10, "t": 0, "b": 0}, showlegend=False)
    map_fig.update_geos(projection_scale=zoom, projection_type="orthographic", showcountries=True)
    return map_fig


def create_check_options(values: list) -> list:
    """
    Creates dash HTML options for checklist based on values.
    :param values: List of values to create options for.
    :return: List of dash HTML options.
    """
    options = []
    for val in values:
        options.append(
            {"label": [html.I(className="bi bi-lock"), html.Span(val)],
             "value": val})
    return options


def compute_percent_change(context: pd.Series, presc: pd.Series) -> float:
    """
    Computes percent land use change from context to presc
    :param context: Context land use data
    :param presc: Prescribed land use data
    :return: Percent land use change
    """
    diffs = presc[RECO_COLS] - context[RECO_COLS]
    change = diffs[diffs > 0].sum()
    percent_changed = change/ context[LAND_USE_COLS].sum()

    return percent_changed


def _create_hovertext(labels: list, parents: list, values: list, title: str) -> list:
    """
    Helper function that formats the hover text for the treemap to be 2 decimals.
    :param labels: Labels according to treemap format.
    :param parents: Parents for each label according to treemap format.
    :param values: Values for each label according to treemap format.
    :param title: Title of treemap, root node's name.
    :return: List of hover text strings.
    """
    hovertext = []
    for i, label in enumerate(labels):
        v = values[i] * 100
        # Get value of parent or 100 if parent is ''
        parent_v = values[labels.index(parents[i])] * 100 if parents[i] != '' else values[0] * 100
        if parents[i] == '':
            hovertext.append(f"{label}: {v:.2f}%")
        elif parents[i] == title:
            hovertext.append(f"{label}<br>{v:.2f}% of {title}")
        else:
            hovertext.append(f"{label}<br>{v:.2f}% of {title}<br>{(v/parent_v)*100:.2f}% of {parents[i]}")

    return hovertext


def create_treemap(data=pd.Series, type_context=True, year=2021) -> go.Figure:
    """
    :param data: Pandas series of land use data
    :param type_context: If the title should be context or prescribed
    :return: Treemap figure
    """
    title = f"Context in {year}" if type_context else f"Prescribed for {year+1}"

    tree_params = {
        "branchvalues": "total",
        "sort": False,
        "texttemplate": "%{label}<br>%{percentRoot:.2%}",
        "hoverinfo": "label+percent root+percent parent",
        "root_color": "lightgrey"
    }

    labels, parents, values = None, None, None

    if data.empty:
        labels = [title]
        parents = [""]
        values = [1]

    else:
        total = data[LAND_USE_COLS].sum()
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
                "Urban",
                "Fields", "pastr", "range"]
        parents = ["", title,
                title, "Crops", "Crops", "C3", "C3", "C3", "C4", "C4",
                title, "Primary Vegetation", "Primary Vegetation",
                title, "Secondary Vegetation", "Secondary Vegetation",
                title,
                title, "Fields", "Fields"]

        values =  [total + data["nonland"], data["nonland"],
                    crops, c3, c4, data["c3ann"], data["c3nfx"], data["c3per"], data["c4ann"], data["c4per"],
                    primary, data["primf"], data["primn"],
                    secondary, data["secdf"], data["secdn"],
                    data["urban"],
                    fields, data["pastr"], data["range"]]

        tree_params["customdata"] = _create_hovertext(labels, parents, values, title)
        tree_params["hovertemplate"] = "%{customdata}<extra></extra>"
 
    assert len(labels) == len(parents)
    assert len(parents) == len(values)

    fig = go.Figure(
        go.Treemap(
            labels = labels,
            parents = parents,
            values = values,
            **tree_params
        )
    )
    colors = px.colors.qualitative.Plotly
    fig.update_layout(
        treemapcolorway = [colors[1], colors[4], colors[2], colors[7], colors[3], colors[0]],
        margin={"t": 0, "b": 0, "l": 10, "r": 10}
    )
    return fig


def create_pie(data=pd.Series, type_context=True, year=2021) -> go.Figure:
    """
    :param data: Pandas series of land use data
    :param type_context: If the title should be context or prescribed
    :return: Pie chart figure
    """

    values = None

    # Sum for case where all zeroes, which allows us to display pie even when presc is reset
    if data.empty or data.sum() == 0:
        values = [0 for _ in range(len(CHART_COLS))]
        values[-1] = 1

    else:
        values = data[CHART_COLS].tolist()

    assert(len(values) == len(CHART_COLS))

    title = f"Context in {year}" if type_context else f"Prescribed for {year+1}"

    p = px.colors.qualitative.Plotly
    ps = px.colors.qualitative.Pastel1
    d = px.colors.qualitative.Dark24
    #['c3ann', 'c3nfx', 'c3per', 'c4ann', 'c4per', 'pastr', 'primf', 'primn', 
    # 'range', 'secdf', 'secdn', 'urban', 'nonland]
    colors = [p[4], d[8], ps[4], p[9], ps[5], p[0], p[2], d[14], p[5], p[7], d[2], p[3], p[1]]
    fig = go.Figure(
        go.Pie(
            values = values,
            labels = CHART_COLS,
            textposition = "inside",
            sort = False,
            marker_colors = colors,
            hovertemplate = "%{label}<br>%{value}<br>%{percent}<extra></extra>",
            title = title
        )
    )

    if type_context:
        fig.update_layout(showlegend=False)
        # To make up for the hidden legend
        fig.update_layout(margin={"t": 50, "b": 50, "l": 50, "r": 50})

    else:
        fig.update_layout(margin={"t": 0, "b": 0, "l": 0, "r": 0})

    return fig


def create_pareto(pareto_df: pd.DataFrame, presc_id: int) -> go.Figure:
    """
    :param pareto_df: Pandas data frame containing the pareto front
    :param presc_id: The currently selected prescriptor id
    :return: A pareto plot figure
    """
    fig = go.Figure(
            go.Scatter(
                x=pareto_df['change'] * 100,
                y=pareto_df['ELUC'],
                # marker='o',
            )
        )
    # Highlight the selected prescriptor
    presc_df = pareto_df[pareto_df["id"] == presc_id]
    fig.add_scatter(x=presc_df['change'] * 100,
                    y=presc_df['ELUC'],
                    marker={
                        "color": 'red',
                        "size": 10
                    })
    # Name axes and hide legend
    fig.update_layout(xaxis_title={"text": "Change (%)"},
                      yaxis_title={"text": 'ELUC (tC/ha/yr)'},
                      showlegend=False,
                      title="Prescriptors",
                      )
    fig.update_traces(hovertemplate="Average Change: %{x} <span>&#37;</span>"
                                    "<br>"
                                    " Average ELUC: %{y} tC/ha/yr<extra></extra>")
    return fig
