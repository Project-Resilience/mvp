"""
Utility functions for the demo application.
"""
from dash import html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app.constants as app_constants
from data import constants

from prescriptors.prescriptor_manager import PrescriptorManager
from prescriptors.nsga2.land_use_prescriptor import LandUsePrescriptor

from predictors.predictor import Predictor
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor
from predictors.percent_change.percent_change_predictor import PercentChangePredictor
from predictors.sklearn.sklearn_predictor import LinearRegressionPredictor, RandomForestPredictor

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

def create_map(df: pd.DataFrame, zoom=10, color_idx = None) -> go.Figure:
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
        size_max=10
    )

    map_fig.update_layout(margin={"l": 0, "r": 10, "t": 0, "b": 0}, showlegend=False)
    map_fig.update_geos(projection_scale=zoom,
                        projection_type="orthographic",
                        showcountries=True,
                        fitbounds="locations")
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

def context_presc_to_df(context: pd.Series, presc: pd.Series) -> pd.DataFrame:
    """
    Takes a context with all columns and a presc with RECO_COLS and returns an updated context actions df.
    This df takes the difference between the RECO_COLS in presc and context and sets the DIFF_RECO_COLS to that.
    """
    diff = presc - context[constants.RECO_COLS]
    diff = diff.rename({col: f"{col}_diff" for col in diff.index})
    context_actions = diff.combine_first(context[constants.CAO_MAPPING["context"]])
    context_actions_df = pd.DataFrame([context_actions])
    context_actions_df[constants.NO_CHANGE_COLS] = 0 # TODO: I'm not entirely sure why this line is necessary
    return context_actions_df

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
        val = values[i] * 100
        # Get value of parent or 100 if parent is '' or 0
        if parents[i] == '' or values[labels.index(parents[i])] == 0:
            parent_v = values[0] * 100
        else:
            parent_v = values[labels.index(parents[i])] * 100
        if parents[i] == '':
            hovertext.append(f"{label}: {val:.2f}%")
        elif parents[i] == title:
            hovertext.append(f"{label}<br>{val:.2f}% of {title}")
        else:
            hovertext.append(f"{label}<br>{val:.2f}% of {title}<br>{(val/parent_v)*100:.2f}% of {parents[i]}")

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
        total = data[constants.LAND_USE_COLS].sum()
        primary = data[app_constants.PRIMARY].sum()
        secondary = data[app_constants.SECONDARY].sum()
        fields = data[app_constants.FIELDS].sum()

        labels = [title, "Nonland",
                "Crops", 
                "Primary Vegetation", "primf", "primn", 
                "Secondary Vegetation", "secdf", "secdn",
                "Urban",
                "Fields", "pastr", "range"]
        parents = ["", title,
                title,
                title, "Primary Vegetation", "Primary Vegetation",
                title, "Secondary Vegetation", "Secondary Vegetation",
                title,
                title, "Fields", "Fields"]

        values =  [total + data["nonland"], data["nonland"],
                    data["crop"],
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
        values = [0 for _ in range(len(app_constants.CHART_COLS))]
        values[-1] = 1

    else:
        values = data[app_constants.CHART_COLS].tolist()

    assert(len(values) == len(app_constants.CHART_COLS))

    title = f"Context in {year}" if type_context else f"Prescribed for {year+1}"

    # Attempt to match the colors from the treemap
    plo = px.colors.qualitative.Plotly
    dar = px.colors.qualitative.Dark24
    #['crop', 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban', 'nonland]
    colors = [plo[4], plo[0], plo[2], dar[14], plo[5], plo[7], dar[2], plo[3], plo[1]]
    fig = go.Figure(
        go.Pie(
            values = values,
            labels = app_constants.CHART_COLS,
            textposition = "inside",
            sort = False,
            marker_colors = colors,
            hovertemplate = "%{label}<br>%{value}<br>%{percent}<extra></extra>",
            title = title
        )
    )

    # Remove the legend from the left plot so that we don't have 2
    if type_context:
        fig.update_layout(showlegend=False)

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
                      yaxis_title={"text": 'ELUC (tC/ha)'},
                      showlegend=False,
                      title="Prescriptors",
                      )
    fig.update_traces(hovertemplate="Average Change: %{x} <span>&#37;</span>"
                                    "<br>"
                                    " Average ELUC: %{y} tC/ha<extra></extra>")
    return fig

def load_prescriptors() -> tuple[list[str], PrescriptorManager]:
    """
    Loads in prescriptors from disk, downloads from HuggingFace first if needed.
    TODO: Currently hard-coded to load specific prescriptors from pareto path.
    :return: dict of prescriptor name -> prescriptor object.
    """
    prescriptors = {}
    pareto_df = pd.read_csv(app_constants.PARETO_CSV_PATH)
    pareto_df = pareto_df.sort_values(by="change")
    for cand_id in pareto_df["id"]:
        cand_path = f"danyoung/eluc-{cand_id}"
        cand_local_dir = app_constants.PRESCRIPTOR_PATH / cand_path.replace("/", "--")
        prescriptors[cand_id] = LandUsePrescriptor.from_pretrained(cand_path, local_dir=cand_local_dir)

    change_predictor = PercentChangePredictor()
    prescriptor_manager = PrescriptorManager(prescriptors, {"change": change_predictor})

    return prescriptor_manager

def load_predictors() -> dict[str, Predictor]:
    """
    Loads in predictors from disk, downloads from HuggingFace first if needed.
    TODO: Currently hard-coded to load specific predictors. We need to make this able to handle any amount!
    :return: dict of predictor name -> predictor object.
    """
    predictors = {}
    nn_path = "danyoung/eluc-global-nn"
    nn_local_dir = app_constants.PREDICTOR_PATH / nn_path.replace("/", "--")
    linreg_path = "danyoung/eluc-global-linreg"
    linreg_local_dir = app_constants.PREDICTOR_PATH / linreg_path.replace("/", "--")
    rf_path = "danyoung/eluc-global-rf"
    rf_local_dir = app_constants.PREDICTOR_PATH / rf_path.replace("/", "--")
    global_nn = NeuralNetPredictor.from_pretrained(nn_path,
                                                   local_dir=nn_local_dir)
    global_linreg = LinearRegressionPredictor.from_pretrained(linreg_path,
                                                              local_dir=linreg_local_dir)
    global_rf = RandomForestPredictor.from_pretrained(rf_path,
                                                      local_dir=rf_local_dir)

    predictors["Global Neural Network"] = global_nn
    predictors["Global Linear Regression"] = global_linreg
    predictors["Global Random Forest"] = global_rf

    return predictors
