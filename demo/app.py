from math import isclose

import numpy as np
import pandas as pd
from dash import ALL
from dash import Dash
from dash import Input
from dash import Output
from dash import State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

from Predictor import XGBoostPredictor
from Predictor import LSTMPredictor
from Prescriptor import Prescriptor
from constants import ALL_LAND_USE_COLS
from constants import CHART_COLS
from constants import CONTEXT_COLUMNS
from constants import ALL_DIFF_LAND_USE_COLS
from constants import LAND_USE_COLS
from constants import PRESCRIPTOR_LIST
from constants import DEFAULT_PRESCRIPTOR_IDX
from constants import PREDICTOR_LIST
from constants import SLIDER_PRECISION
from constants import MAP_COORDINATE_DICT
from constants import CO2_JFK_GVA
from constants import HISTORY_SIZE
from constants import PRESCRIPTOR_COLS
from constants import CO2_PERSON
from constants import CHART_TYPES
from constants import PARETO_CSV_PATH
from utils import add_nonland
from utils import create_map
from utils import create_check_options
from utils import compute_percent_change
from utils import create_treemap
from utils import create_pie
from utils import create_pareto

app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
           prevent_initial_callbacks="initial_duplicate")

# TODO: should we load all our data into a store?
# This seems more secure.
df = pd.read_csv("../data/gcb/processed/gb_br_ch_eluc.csv", index_col=0)
#df = pd.read_csv("../data/gcb/processed/uk_eluc.csv")
pareto_df = pd.read_csv(PARETO_CSV_PATH)

# Cells
GRID_STEP = 0.25
min_lat = df["i_lat"].min()
max_lat = df["i_lat"].max()
min_lon = df["i_lon"].min()
max_lon = df["i_lon"].max()
min_year = df["time"].min()
max_year = df["time"].max()

lat_list = list(np.arange(min_lat, max_lat + GRID_STEP, GRID_STEP))
lon_list = list(np.arange(min_lon, max_lon + GRID_STEP, GRID_STEP))

present = df[df["time"] == 2021]
map_fig = create_map(present, 54.5, -2.5, 20)

# Legend examples come from https://hess.copernicus.org/preprints/hess-2021-247/hess-2021-247-ATC3.pdf
legend_div = html.Div(
    style={},
    children = [
        dcc.Markdown('''
### Land Use Types

Primary: Vegetation that is untouched by humans

    - primf: Primary forest
    - primn: Primary nonforest vegetation

    
Secondary: Vegetation that has been touched by humans

    - secdf: Secondary forest
    - secdn: Secondary nonforest vegetation

Urban

Crop

    - c3ann: Annual C3 crops (e.g. wheat)
    - c4ann: Annual C4 crops (e.g. maize)
    - c3per: Perennial C3 crops (e.g. banana)
    - c4per: Perennial C4 crops (e.g. sugarcane)
    - c3nfx: Nitrogen fixing C3 crops (e.g. soybean)

Pasture

    - pastr: Managed pasture land
    - range: Natural grassland/savannah/desert/etc.
    ''')
    ]
)

context_div = html.Div(
    style={'display': 'grid',
           'grid-template-columns': 'auto 1fr', 'grid-template-rows': 'auto auto auto auto',
           'position': 'absolute', 'bottom': '0'},
    children=[
        html.P("Region", style={'grid-column': '1', 'grid-row': '1', 'padding-right': '10px'}),
        dcc.Dropdown(
            id="loc-dropdown",
            options=list(MAP_COORDINATE_DICT.keys()),
            value=list(MAP_COORDINATE_DICT.keys())[0],
            style={'grid-column': '2', 'grid-row': '1', 'width': '75%', 'justify-self': 'left', 'margin-top': '-3px'}
        ),
        html.P("Lat", style={'grid-column': '1', 'grid-row': '2', 'padding-right': '10px'}),
        dcc.Dropdown(
            id='lat-dropdown',
            options=lat_list,
            placeholder="Select a latitude",
            value=51.625,
            style={'grid-column': '2', 'grid-row': '2', 'width': '75%', 'justify-self': 'left', 'margin-top': '-3px',}
        ),
        html.P("Lon", style={'grid-column': '1', 'grid-row': '3', 'padding-right': '10px'}),
        dcc.Dropdown(
            id='lon-dropdown',
            options=lon_list,
            placeholder="Select a longitude",
            value=-3.375,
            style={'grid-column': '2', 'grid-row': '3', 'width': '75%', 'justify-self': 'left', 'margin-top': '-3px'}
        ),
        html.P("Year ", style={'grid-column': '1', 'grid-row': '4', 'margin-right': '10px'}),
        html.Div([
            dcc.Input(
                id="year-input",
                type="number",
                value=2021,
                debounce=True
            ),
            dcc.Tooltip(f"Year must be between {min_year} and {max_year}."),
        ], style={'grid-column': '2', 'grid-row': '4', 'width': '75%', 'justify-self': 'left', 'margin-top': '-3px'}),
    ]
)

presc_select_div = html.Div([
    html.P("Minimize change", style={"grid-column": "1"}),
    html.Div([
        dcc.Slider(id='presc-select',
                min=0, max=len(PRESCRIPTOR_LIST)-1, step=1,
                value=DEFAULT_PRESCRIPTOR_IDX,
                included=False,
                marks={i : "" for i in range(len(PRESCRIPTOR_LIST))})
    ], style={"grid-column": "2", "width": "100%", "margin-top": "8px"}),
    html.P("Minimize ELUC", style={"grid-column": "3", "padding-right": "10px"}),
    html.Button("Prescribe", id='presc-button', n_clicks=0, style={"grid-column": "4", "margin-top": "-10px"}),
    html.Button("View Pareto", id='pareto-button', n_clicks=0, style={"grid-column": "5", "margin-top": "-10px"}),
    dbc.Modal(
            [
                dbc.ModalHeader("Pareto front"),
                dcc.Graph(id='pareto-fig', figure=create_pareto(pareto_df=pareto_df,
                                                                presc_id=PRESCRIPTOR_LIST[DEFAULT_PRESCRIPTOR_IDX])),
            ],
            id="pareto-modal",
            is_open=False,
        ),
], style={"display": "grid", "grid-template-columns": "auto 1fr auto auto", "width": "100%", "align-content": "center"})

chart_select_div = dcc.Dropdown(
    options=CHART_TYPES,
    id="chart-select",
    value=CHART_TYPES[0],
    clearable=False
)

check_options = create_check_options(LAND_USE_COLS)
checklist_div = html.Div([
    dcc.Checklist(check_options, id="locks", inputStyle={"margin-bottom": "30px"})
])

sliders_div = html.Div([
    html.Div([
        #html.P(col, style={"grid-column": "1"}),
        html.Div([
            dcc.Slider(
                min=0,
                max=1,
                step=SLIDER_PRECISION,
                value=0,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
                id={"type": "presc-slider", "index": f"{col}-slider"}
            )
        ], style={"grid-column": "1", "width": "100%", "margin-top": "8px"}),
        dcc.Input(
            value="0%",
            type="text", 
            disabled=True,
            id={"type": "slider-value", "index": f"{col}-value"},
            style={"grid-column": "2", "text-align": "right", "margin-top": "-5px"}),
    ], style={"display": "grid", "grid-template-columns": "1fr 15%"}) for col in LAND_USE_COLS]
)

frozen_cols = [col for col in CHART_COLS if col not in LAND_USE_COLS]
frozen_div = html.Div([
    dcc.Input(
        value=f"{col}: 0.00%",
        type="text",
        disabled=True,
        id={"type": "frozen-input", "index": f"{col}-frozen"}) for col in frozen_cols
])

predict_div = html.Div([
    dcc.Dropdown(PREDICTOR_LIST, PREDICTOR_LIST[0], id="pred-select", style={"width": "200px"}),
    html.Button("Predict", id='predict-button', n_clicks=0,),
    html.Label("Predicted ELUC:", style={'padding-left': '10px'}),
    dcc.Input(
        value="",
        type="text",
        disabled=True,
        id="predict-eluc",
    ),
    html.Label("tC/ha/yr", style={'padding-left': '2px'}),
    html.Label("Land Change:", style={'padding-left': '10px'}),
    dcc.Input(
        value="",
        type="text",
        disabled=True,
        id="predict-change",
    ),
    html.Label("%", style={'padding-left': '2px'}),
], style={"display": "flex", "flex-direction": "row", "width": "90%", "align-items": "center"})

inline_block = {"display": "inline-block", "padding-right": "10px"}
trivia_div = html.Div([
    html.Div(className="parent", children=[
        html.P("Total emissions reduced from this land use change over a year: ", className="child", style=inline_block),
        html.P(id="total-em", style={"font-weight": "bold"}|inline_block)
    ]),
    html.Div(className="parent", children=[
        html.I(className="bi bi-airplane", style=inline_block),
        html.P("Flight emissions from flying JFK to Geneva: ", className="child", style=inline_block),
        html.P(f"{CO2_JFK_GVA} tonnes CO2", style={"font-weight": "bold"}|inline_block)
    ]),
    html.Div(className="parent", children=[
        html.I(className="bi bi-airplane", style=inline_block),
        html.P("Plane tickets mitigated: ", className="child", style=inline_block),
        html.P(id="tickets", style={"font-weight": "bold"}|inline_block)
    ]),
    html.Div(className="parent", children=[
        html.I(className="bi bi-person", style=inline_block),
        html.P("Total yearly carbon emissions of average world citizen: ", className="child", style=inline_block),
        html.P(f"{CO2_PERSON} tonnes CO2", style={"font-weight": "bold"}|inline_block)
    ]),
    html.Div(className="parent", children=[
        html.I(className="bi bi-person", style=inline_block),
        html.P("Number of peoples' carbon emissions mitigated from this change : ", className="child", style=inline_block),
        html.P(id="people", style={"font-weight": "bold"}|inline_block)
    ]),
    html.P("(Sources: https://flightfree.org/flight-emissions-calculator https://scied.ucar.edu/learning-zone/climate-solutions/carbon-footprint)", style={"font-size": "10px"})
])

references_div = html.Div([
    html.Div(className="parent", children=[
        html.P("ELUC data provided by the BLUE model  ",
               className="child", style=inline_block),
        html.A("(BLUE: Bookkeeping of land use emissions)", href="https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014GB004997\n"),
    ]),
    html.Div(className="parent", children=[
        html.P("Land use change data provided by the LUH2 project",
               className="child", style=inline_block),
        html.A("(LUH2: Land Use Harmonization 2)", href="https://luh.umd.edu/\n"),
    ]),
    html.Div(className="parent", children=[
        html.P("Setup is described in Appendix C2.1 of the GCB 2022 report",
               className="child", style=inline_block),
        html.A("(Global Carbon Budget 2022 report)", href="https://essd.copernicus.org/articles/14/4811/2022/#section10/\n"),
    ]),
    html.Div(className="parent", children=[
        html.P("The Global Carbon Budget report assesses the global CO2 budget for the Intergovernmental Panel on Climate Change",
               className="child", style=inline_block),
        html.A("(IPCC)", href="https://www.ipcc.ch/\n"),
    ]),
])


@app.callback(
    Output("pareto-modal", "is_open"),
    Output("pareto-fig", "figure"),
    [Input("pareto-button", "n_clicks")],
    [State("pareto-modal", "is_open")],
    [State("presc-select", "value")],
)
def toggle_modal(n, is_open, presc_idx):
    """
    Toggles pareto modal.
    :param n: Number of times button has been clicked.
    :param is_open: Whether the modal is open.
    :param presc_idx: The index of the prescriptor to show.
    :return: The new state of the modal and the figure to show.
    """
    fig = create_pareto(pareto_df, PRESCRIPTOR_LIST[presc_idx])
    if n:
        return not is_open, fig
    return is_open, fig


@app.callback(
    Output("lat-dropdown", "value"),
    Output("lon-dropdown", "value"),
    Input("map", "clickData"),
    prevent_initial_call=True
)
def click_map(click_data):
    """
    Selects context when point on map is clicked.
    :param click_data: Input data from click action.
    :return: The new longitude and latitude to put into the dropdowns.
    """
    return click_data["points"][0]["lat"], click_data["points"][0]["lon"]

@app.callback(
    Output("map", "figure"),
    Input("loc-dropdown", "value"),
    Input("year-input", "value"),
    Input("context-store", "data"),
    prevent_initial_call=True
)
def update_map(location, year, context):
    """
    Updates map data behind the scenes when year is clicked.
    Changes focus when region is selected.
    :param location: The name of the country selected from the dropdown.
    :param year: The selected year.
    :return: A newly created map.
    """
    coord_dict = MAP_COORDINATE_DICT[location]
    data = df[df["time"] == year]
    data = data.reset_index(drop=True)
    idx = None
    if context:
        context_df = pd.DataFrame.from_records(context)
        lat = context_df["lat"].iloc[0]
        lon = context_df["lon"].iloc[0]
        idx = list(data.index[(data["lat"] == lat) & (data["lon"] == lon)])[0]

    return create_map(data, coord_dict["lat"], coord_dict["lon"], coord_dict["zoom"], idx)

@app.callback(
    Output("context-store", "data"),
    Output("history-store", "data"),
    Output({"type": "frozen-input", "index": ALL}, "value"),
    Output({"type": "presc-slider", "index": ALL}, "value"),
    Output({"type": "presc-slider", "index": ALL}, "max"),
    Input("lat-dropdown", "value"),
    Input("lon-dropdown", "value"),
    Input("year-input", "value")
)
def select_context(lat, lon, year):
    """
    Loads context in from lat/lon/time. Updates context/history stores and frozen inputs (primf, etc.).
    Also resets prescription sliders to 0 to avoid confusion.
    Also sets prescription sliders' max values to 1 - nonland - primf - primn to avoid negative values.
    :param lat: Selected latitude.
    :param lon: Selected longitude.
    :param year: Selected year.
    :return: Context/history data to store, frozen values, slider values, and slider max.
    """
    context = df[(df['i_lat'] == lat) & (df['i_lon'] == lon) & (df['time'] == year)]
    history = df[(df['i_lat'] == lat) & (df['i_lon'] == lon) & (df['time'] < year) & (df['time'] >= year-HISTORY_SIZE)]

    chart_df = add_nonland(context[ALL_LAND_USE_COLS])

    frozen = chart_df[frozen_cols].iloc[0].tolist()
    #frozen = round_list(frozen)
    frozen = [f"{frozen_cols[i]}: {frozen[i]*100:.2f}%" for i in range(len(frozen_cols))]

    reset = [0 for _ in LAND_USE_COLS]

    max_val = chart_df[LAND_USE_COLS].sum(axis=1).iloc[0]
    maxes = [max_val for _ in range(len(LAND_USE_COLS))]

    return context.to_dict("records"), history.to_dict("records"), frozen, reset, maxes

@app.callback(
    Output("context-fig", "figure"),
    Input("chart-select", "value"),
    Input("context-store", "data"),
    State("year-input", "value")
)
def update_context_chart(chart_type, context, year):
    """
    Updates context chart when context store is updated or chart type is changed.
    :param chart_type: String input from chart select dropdown.
    :param context: Context data from store.
    :param year: Selected year.
    :return: New figure type selected by chart_type with data context.
    """
    context_df = pd.DataFrame.from_records(context)[CONTEXT_COLUMNS]
    chart_df = add_nonland(context_df[ALL_LAND_USE_COLS])

    assert chart_type in ("Treemap", "Pie Chart")

    if chart_type == "Treemap":
        return create_treemap(chart_df.iloc[0], type_context=True, year=year)
    
    return create_pie(chart_df.iloc[0], type_context=True, year=year)

@app.callback(
    Output({"type": "presc-slider", "index": ALL}, "value", allow_duplicate=True),
    Input("presc-button", "n_clicks"),
    State("presc-select", "value"),
    State("context-store", "data"),
    prevent_initial_call=True
)
def select_prescriptor(n_clicks, presc_idx, context):
    """
    Selects prescriptor, runs on context, updates sliders.
    :param n_clicks: Unused number of times button has been clicked.
    :param presc_idx: Index of prescriptor in PRESCRIPTOR_LIST to load.
    :param context: Context data from store to run prescriptor on.
    :return: Updated slider values.
    """
    if context is not None:
        presc_id = PRESCRIPTOR_LIST[presc_idx]
        prescriptor = Prescriptor(presc_id)
        context_df = pd.DataFrame.from_records(context)[PRESCRIPTOR_COLS]
        prescribed = prescriptor.run_prescriptor(context_df)

        # TODO: Hacking this together because c4per isn't prescribed
        prescribed["c4per"] = 0
        return prescribed[LAND_USE_COLS].iloc[0].tolist()
    
    return None


@app.callback(
    Output("presc-store", "data"),
    Output({"type": "slider-value", "index": ALL}, "value"),
    Output("sum-warning", "children"),
    Output("predict-change", "value"),
    Input({"type": "presc-slider", "index": ALL}, "value"),
    State("context-store", "data"),
    State("locks", "value"),
    prevent_initial_call=True
)
def store_prescription(sliders, context, locked):
    """
    Stores slider values in store and displays them next to sliders. Computes land change percent for output.
    Warns user if values don't sum to 1.
    :param sliders: Slider values to store.
    :param context: Context store to compute if prescription sums to land use in context.
    :param locked: Locked columns to check for warning.
    :return: Stored slider values, slider values to display, warning if necessary, land change percent.
    """
    context_df = pd.DataFrame.from_records(context)[CONTEXT_COLUMNS]
    presc = pd.DataFrame([sliders], columns=LAND_USE_COLS)
    vals = presc.iloc[0].tolist()

    warnings = []
    # Check if prescriptions sum to 1
    # TODO: Are we being precise enough?
    new_sum = presc.sum(axis=1).iloc[0]
    old_sum = context_df[LAND_USE_COLS].sum(axis=1).iloc[0]
    if not isclose(new_sum, old_sum, rel_tol=1e-7):
        warnings.append(html.P(f"WARNING: Please make sure prescriptions sum to: {str(old_sum * 100)} instead of {str(new_sum * 100)} by clicking \"Sum to 100\""))

    # Check if sum of locked prescriptions are > sum(land use)
    # TODO: take a look at this logic.
    if locked and presc[locked].sum(axis=1).iloc[0] > old_sum:
        warnings.append(html.P("WARNING: Sum of locked prescriptions is greater than sum of land use. Please reduce one before proceeding"))

    # Check if any prescriptions below 0
    if (presc.iloc[0] < 0).any():
        warnings.append(html.P("WARNING: Negative values detected. Please lower the value of a locked slider."))

    # Compute total change
    change = compute_percent_change(context_df, presc)

    return presc.to_dict("records"), [f"{val * 100:.2f}%" for val in vals], warnings, f"{change * 100:.2f}"


@app.callback(
    Output("presc-fig", "figure", allow_duplicate=True),
    Input("chart-select", "value"),
    Input("presc-store", "data"),
    State("context-store", "data"),
    State("year-input", "value"),
    prevent_initial_call=True
)
def update_presc_chart(chart_type, presc, context, year):
    """
    Updates prescription pie from store according to chart type.
    :param chart_type: String input from chart select dropdown.
    :param presc: Prescription data from store.
    :param context: Context data from store.
    :param year: Selected year for title of chart.
    :return: New chart of type chart_type using presc data.
    """
    presc_df = pd.DataFrame.from_records(presc)[LAND_USE_COLS]
    context_df = pd.DataFrame.from_records(context)[CONTEXT_COLUMNS]
    presc_df["primf"] = context_df["primf"]
    presc_df["primn"] = context_df["primn"]

    # Manually calculate nonland from context so that it's not zeroed out by sliders.
    nonland = 1 - context_df[ALL_LAND_USE_COLS].iloc[0].sum()
    nonland = nonland if nonland > 0 else 0
    presc_df["nonland"] = nonland
    chart_df = presc_df[CHART_COLS]

    assert chart_type in ("Treemap", "Pie Chart")

    if chart_type == "Treemap":
        return create_treemap(chart_df.iloc[0], type_context=False, year=year)
    
    return create_pie(chart_df.iloc[0], type_context=False, year=year)


@app.callback(
    Output({"type": "presc-slider", "index": ALL}, "value", allow_duplicate=True),
    Input("sum-button", "n_clicks"),
    State("presc-store", "data"),
    State("context-store", "data"),
    State("locks", "value"),
    prevent_initial_call=True
)
def sum_to_1(n_clicks, presc, context, locked):
    """
    Sets slider values to sum to how much land was used in context.
    Subtracts locked sum from both of these and doesn't adjust them.
    :param n_clicks: Unused number of times button has been clicked.
    :param presc: Prescription data from store.
    :param context: Context data from store.
    :param locked: Which sliders to not consider in calculation.
    :return: Slider values scaled down to fit percentage of land used in context.
    """
    context_df = pd.DataFrame.from_records(context)[LAND_USE_COLS]
    presc_df = pd.DataFrame.from_records(presc)[LAND_USE_COLS]

    old_sum = context_df.sum(axis=1).iloc[0]
    new_sum = presc_df.sum(axis=1).iloc[0]

    # TODO: There is certainly a more elegant way to handle this.
    if locked:
        unlocked = [col for col in LAND_USE_COLS if col not in locked]
        locked_sum = presc_df[locked].sum(axis=1).iloc[0]
        old_sum -= locked_sum
        new_sum -= locked_sum
        # We do this to avoid divide by zero. In the case where new_sum == 0
        # we have all locked columns and/or zero columns so no adjustment is needed
        if new_sum != 0:
            presc_df[unlocked] = presc_df[unlocked].div(new_sum, axis=0).mul(old_sum, axis=0)

    else:
        presc_df = presc_df.div(new_sum, axis=0).mul(old_sum, axis=0)

    # Set all negative values to 0
    presc_df[presc_df < 0] = 0
    return presc_df.iloc[0].tolist()


@app.callback(
    Output("predict-eluc", "value"),
    Output("total-em", "children"),
    Output("tickets", "children"),
    Output("people", "children"),
    Input("predict-button", "n_clicks"),
    State("context-store", "data"),
    State("history-store", "data"),
    State("presc-store", "data"),
    State("pred-select", "value"),
    prevent_initial_call=True
)
def predict(n_clicks, context, history, presc, predictor_name):
    """
    Predicts ELUC from context and prescription stores.
    :param n_clicks: Unused number of times button has been clicked.
    :param context: Context data from store.
    :param history: History data for time series predictors.
    :param presc: Prescription data from store.
    :param predictor_name: String name of predictor to use from dropdown.
    :return: Predicted ELUC and trivia values.
    """
    context_df = pd.DataFrame.from_records(context)
    history_df = pd.DataFrame.from_records(history)
    presc_df = pd.DataFrame.from_records(presc)[LAND_USE_COLS]

    predictor = None
    prediction, change = 0, 0

    if predictor_name == "XGBoost":
        predictor = XGBoostPredictor()
        prediction = predictor.run_predictor(context_df[CONTEXT_COLUMNS], presc_df)

    elif predictor_name == "LSTM":
        predictor = LSTMPredictor()
        context_merged = pd.concat([history_df, context_df], axis=0)
        # Sanity check
        context_merged = context_merged.sort_values(by="time", ascending=True)
        # TODO: This is gross, clean it up.
        prediction = predictor.run_predictor(context_merged[CONTEXT_COLUMNS + ALL_DIFF_LAND_USE_COLS], presc_df)

    else:
        return 0, 0, "Model not connected yet"
    
    # Calculate total reduction
    area = context_df["cell_area"].iloc[0]
    total_reduction = prediction * area
    
    return f"{prediction:.4f}", \
        f"{-1 * total_reduction:,.2f} tonnes CO2", \
            f"{-1 * total_reduction // CO2_JFK_GVA:,.0f} tickets", \
                f"{-1 * total_reduction // CO2_PERSON:,.0f} people"


def main():
    global app
    app.title = 'Land Use Optimization'
    app.css.config.serve_locally = False
    # Don't be afraid of the 3rd party URLs: chriddyp is the author of Dash!
    # These two allow us to dim the screen while loading.
    # See discussion with Dash devs here: https://community.plotly.com/t/dash-loading-states/5687
    app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
    app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/brPBPO.css'})

    app.layout = html.Div([
        dcc.Store(id='context-store'),
        dcc.Store(id='history-store'),
        dcc.Store(id='presc-store'),
        dcc.Markdown('''
# Land Use Optimization
This site is for demonstration purposes only.

For a given context cell representing a portion of the earth,
identified by its latitude and longitude coordinates, and a given year:
* What changes can we make to the land usage
* In order to minimize the resulting estimated CO2 emissions in that year ? (Emissions from Land Use Change, ELUC, 
in tons of carbon per hectare per year)
'''),
        dcc.Markdown('''## Context'''),
        html.Div([
            dcc.Graph(id="map", figure=map_fig, style={"grid-column": "1"}),
            html.Div([context_div], style={"grid-column": "2"}),
            html.Div([legend_div], style={"grid-column": "3"})
        ], style={"display": "grid", "grid-template-columns": "auto 1fr auto", 'position': 'relative'}),
        dcc.Markdown('''## Actions'''),
        html.Div([
            html.Div([presc_select_div], style={"grid-column": "1"}),
            html.Div([chart_select_div], style={"grid-column": "2", "margin-top": "-10px", "margin-left": "10px"}),
        ], style={"display": "grid", "grid-template-columns": "45% 15%"}),
        html.Div([
            html.Div(checklist_div, style={"grid-column": "1", "height": "100%"}),
            html.Div(sliders_div, style={'grid-column': '2'}),
            dcc.Graph(id='context-fig', figure=create_treemap(type_context=True), style={'grid-column': '3'}),
            dcc.Graph(id='presc-fig', figure=create_treemap(type_context=False), style={'grid-clumn': '4'})
        ], style={'display': 'grid', 'grid-template-columns': 'auto 40% 1fr 1fr', "width": "100%"}),
        html.Div([
            frozen_div,
            html.Button("Sum to 100%", id='sum-button', n_clicks=0),
            html.Div(id='sum-warning')
        ]),
        dcc.Markdown('''## Outcomes'''),
        predict_div,
        dcc.Markdown('''## Trivia'''),
        trivia_div,
        dcc.Markdown('''## References'''),
        references_div
    ], style={'padding-left': '10px'},)

    app.run_server(debug=True)


if __name__ == '__main__':
    main()
