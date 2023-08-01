from math import isclose

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL
from dash import MATCH
from dash import Dash
from dash import Input
from dash import Output
from dash import State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

import app.Predictor as Predictor
import app.Prescriptor as Prescriptor
import app.constants as constants
import app.utils as utils

app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
           prevent_initial_callbacks="initial_duplicate")
server = app.server

# Remove "Server" http header for security (by obscurity) 
# Note: this is not a security measure, it's just a way to hide the fact that we're using Flask
# Note2: Using "del" to remove the header entirely doesn't work here. Something in Flask/Dash just adds it back.
@server.after_request
def remove_headers(response):
    response.headers['Server'] = ''
    return response

# TODO: should we load all our data into a store?
# This seems more secure.
df = pd.read_csv(constants.DATA_FILE_PATH, index_col=constants.INDEX_COLS)
#df = pd.read_csv("../data/gcb/processed/uk_eluc.csv")
pareto_df = pd.read_csv(constants.PARETO_CSV_PATH)
# We have to reverse for some reason?
prescriptor_list = list(pareto_df["id"])
prescriptor_list.reverse()

# Cells
GRID_STEP = 0.25
min_lat = df.index.get_level_values("lat").min()
max_lat = df.index.get_level_values("lat").max()
min_lon = df.index.get_level_values("lon").min()
max_lon = df.index.get_level_values("lon").max()
min_time = df.index.get_level_values("time").min()
max_time = df.index.get_level_values("time").max()

lat_list = list(np.arange(min_lat, max_lat + GRID_STEP, GRID_STEP))
lon_list = list(np.arange(min_lon, max_lon + GRID_STEP, GRID_STEP))

map_fig = go.Figure()

#TODO: Is this allowed?
random_forest_predictor = Predictor.RandomForestPredictor()

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
            options=list(constants.MAP_COORDINATE_DICT.keys()),
            value=list(constants.MAP_COORDINATE_DICT.keys())[0],
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
            dcc.Tooltip(f"Year must be between {min_time} and {max_time}."),
        ], style={'grid-column': '2', 'grid-row': '4', 'width': '75%', 'justify-self': 'left', 'margin-top': '-3px'}),
    ]
)

presc_select_div = html.Div([
    html.P("Minimize change", style={"grid-column": "1"}),
    html.Div([
        dcc.Slider(id='presc-select',
                min=0, max=len(prescriptor_list)-1, step=1,
                value=constants.DEFAULT_PRESCRIPTOR_IDX,
                included=False,
                marks={i : "" for i in range(len(prescriptor_list))})
    ], style={"grid-column": "2", "width": "100%", "margin-top": "8px"}),
    html.P("Minimize ELUC", style={"grid-column": "3", "padding-right": "10px"}),
    html.Button("Prescribe", id='presc-button', n_clicks=0, style={"grid-column": "4", "margin-top": "-10px"}),
    html.Button("View Pareto", id='pareto-button', n_clicks=0, style={"grid-column": "5", "margin-top": "-10px"}),
    dbc.Modal(
            [
                dbc.ModalHeader("Pareto front"),
                dcc.Graph(id='pareto-fig', figure=utils.create_pareto(pareto_df=pareto_df,
                                                                presc_id=prescriptor_list[constants.DEFAULT_PRESCRIPTOR_IDX])),
            ],
            id="pareto-modal",
            is_open=False,
        ),
], style={"display": "grid", "grid-template-columns": "auto 1fr auto auto", "width": "100%", "align-content": "center"})

chart_select_div = dcc.Dropdown(
    options=constants.CHART_TYPES,
    id="chart-select",
    value=constants.CHART_TYPES[0],
    clearable=False
)

check_options = utils.create_check_options(constants.RECO_COLS)
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
                step=constants.SLIDER_PRECISION,
                value=0,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
                id={"type": "presc-slider", "index": f"{col}"}
            )
        ], style={"grid-column": "1", "width": "100%", "margin-top": "8px"}),
        dcc.Input(
            value="0%",
            type="text", 
            disabled=True,
            id={"type": "slider-value", "index": f"{col}"},
            style={"grid-column": "2", "text-align": "right", "margin-top": "-5px"}),
    ], style={"display": "grid", "grid-template-columns": "1fr 15%"}) for col in constants.RECO_COLS]
)

frozen_div = html.Div([
    dcc.Input(
        value=f"{col}: 0.00%",
        type="text",
        disabled=True,
        id={"type": "frozen-input", "index": f"{col}-frozen"}) for col in constants.NO_CHANGE_COLS + ["nonland"]
])

predict_div = html.Div([
    dcc.Dropdown(constants.PREDICTOR_LIST, constants.PREDICTOR_LIST[0], id="pred-select", style={"width": "200px"}),
    html.Button("Predict", id='predict-button', n_clicks=0,),
    html.Label("Predicted ELUC:", style={'padding-left': '10px'}),
    dcc.Input(
        value="",
        type="text",
        disabled=True,
        id="predict-eluc",
    ),
    html.Label("tC/ha", style={'padding-left': '2px'}),
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
        html.P("Total emissions reduced from this land use change: ", className="child", style=inline_block),
        html.P(id="total-em", style={"font-weight": "bold"}|inline_block)
    ]),
    html.Div(className="parent", children=[
        html.I(className="bi bi-airplane", style=inline_block),
        html.P("Flight emissions from flying JFK to Geneva: ", className="child", style=inline_block),
        html.P(f"{constants.CO2_JFK_GVA} tonnes CO2", style={"font-weight": "bold"}|inline_block)
    ]),
    html.Div(className="parent", children=[
        html.I(className="bi bi-airplane", style=inline_block),
        html.P("Plane tickets mitigated: ", className="child", style=inline_block),
        html.P(id="tickets", style={"font-weight": "bold"}|inline_block)
    ]),
    html.Div(className="parent", children=[
        html.I(className="bi bi-person", style=inline_block),
        html.P("Total yearly carbon emissions of average world citizen: ", className="child", style=inline_block),
        html.P(f"{constants.CO2_PERSON} tonnes CO2", style={"font-weight": "bold"}|inline_block)
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
    fig = utils.create_pareto(pareto_df, prescriptor_list[presc_idx])
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
    Input("lat-dropdown", "value"),
    Input("lon-dropdown", "value")
)
def update_map(location, year, lat, lon):
    """
    Updates map data behind the scenes when year is clicked.
    Changes focus when region is selected.
    :param location: The name of the country selected from the dropdown.
    :param year: The selected year.
    :return: A newly created map.
    """
    coord_dict = constants.MAP_COORDINATE_DICT[location]
    data = df.loc[year]
    data = data.copy().reset_index()
    lat_lon = (data["lat"] == lat) & (data["lon"] == lon)
    idx = data[lat_lon].index[0]

    return utils.create_map(data, coord_dict["lat"], coord_dict["lon"], coord_dict["zoom"], idx)


@app.callback(
    Output({"type": "frozen-input", "index": ALL}, "value"),
    Output({"type": "presc-slider", "index": ALL}, "value"),
    Output({"type": "presc-slider", "index": ALL}, "max"),
    Input("lat-dropdown", "value"),
    Input("lon-dropdown", "value"),
    Input("year-input", "value")
)
def set_frozen_reset_sliders(lat, lon, year):
    """
    Resets prescription sliders to 0 to avoid confusion.
    Also sets prescription sliders' max values to 1 - no change cols to avoid negative values.
    :param lat: Selected latitude.
    :param lon: Selected longitude.
    :param year: Selected year.
    :return: Frozen values, slider values, and slider max.
    """
    context = df.loc[year, lat, lon]

    chart_data = utils.add_nonland(context[constants.LAND_USE_COLS])

    frozen_cols = constants.NO_CHANGE_COLS + ["nonland"]
    frozen = chart_data[frozen_cols].tolist()
    frozen = [f"{frozen_cols[i]}: {frozen[i]*100:.2f}%" for i in range(len(frozen_cols))]

    reset = [0 for _ in constants.RECO_COLS]
    
    max_val = chart_data[constants.RECO_COLS].sum()
    maxes = [max_val for _ in range(len(constants.RECO_COLS))]

    return frozen, reset, maxes


@app.callback(
    Output("context-fig", "figure"),
    Input("chart-select", "value"),
    Input("year-input", "value"),
    Input("lat-dropdown", "value"),
    Input("lon-dropdown", "value")
)
def update_context_chart(chart_type, year, lat, lon):
    """
    Updates context chart when context store is updated or chart type is changed.
    :param chart_type: String input from chart select dropdown.
    :param year: Selected context year.
    :param lat: Selected context lat.
    :param lon: Selected context lon.
    :return: New figure type selected by chart_type with data context.
    """
    context = df.loc[year, lat, lon]
    chart_data = utils.add_nonland(context[constants.LAND_USE_COLS])

    assert chart_type in ("Treemap", "Pie Chart")

    if chart_type == "Treemap":
        return utils.create_treemap(chart_data, type_context=True, year=year)
    
    return utils.create_pie(chart_data, type_context=True, year=year)


@app.callback(
    Output({"type": "presc-slider", "index": ALL}, "value", allow_duplicate=True),
    Input("presc-button", "n_clicks"),
    State("presc-select", "value"),
    State("year-input", "value"),
    State("lat-dropdown", "value"),
    State("lon-dropdown", "value"),
    prevent_initial_call=True
)
def select_prescriptor(n_clicks, presc_idx, year, lat, lon):
    """
    Selects prescriptor, runs on context, updates sliders.
    :param n_clicks: Unused number of times button has been clicked.
    :param presc_idx: Index of prescriptor in PRESCRIPTOR_LIST to load.
    :param year: Selected context year.
    :param lat: Selected context lat.
    :param lon: Selected context lon.
    :return: Updated slider values.
    """
    presc_id = prescriptor_list[presc_idx]
    prescriptor = Prescriptor.Prescriptor(presc_id)
    context = df.loc[year, lat, lon][constants.CONTEXT_COLUMNS]
    context_df = pd.DataFrame([context])
    prescribed = prescriptor.run_prescriptor(context_df)
    return prescribed.iloc[0].tolist()
    

@app.callback(
    Output({"type": "slider-value", "index": MATCH}, "value"),
    Input({"type": "presc-slider", "index": MATCH}, "value")
)
def show_slider_value(slider):
    """
    Displays slider values next to sliders.
    :param sliders: Slider values.
    :return: Slider values.
    """
    return f"{slider * 100:.2f}%"


@app.callback(
    Output("sum-warning", "children"),
    Output("predict-change", "value"),
    Input({"type": "presc-slider", "index": ALL}, "value"),
    State("year-input", "value"),
    State("lat-dropdown", "value"),
    State("lon-dropdown", "value"),
    State("locks", "value"),
    prevent_initial_call=True
)
def compute_land_change(sliders, year, lat, lon, locked):
    """
    Computes land change percent for output.
    Warns user if values don't sum to 1.
    :param sliders: Slider values to store.
    :param year: Selected context year.
    :param lat: Selected context lat.
    :param lon: Selected context lon.
    :param locked: Locked columns to check for warning.
    :return: Warning if necessary, land change percent.
    """
    context = df.loc[year, lat, lon][constants.LAND_USE_COLS]
    presc = pd.Series(sliders, index=constants.RECO_COLS)

    warnings = []
    # Check if prescriptions sum to 1
    # TODO: Are we being precise enough?
    new_sum = presc.sum()
    old_sum = context[constants.RECO_COLS].sum()
    if not isclose(new_sum, old_sum, rel_tol=1e-7):
        warnings.append(html.P(f"WARNING: Please make sure prescriptions sum to: {str(old_sum * 100)} instead of {str(new_sum * 100)} by clicking \"Sum to 100\""))

    # Check if sum of locked prescriptions are > sum(land use)
    # TODO: take a look at this logic.
    if locked and presc[locked].sum() > old_sum:
        warnings.append(html.P("WARNING: Sum of locked prescriptions is greater than sum of land use. Please reduce one before proceeding"))

    # Check if any prescriptions below 0
    if (presc < 0).any():
        warnings.append(html.P("WARNING: Negative values detected. Please lower the value of a locked slider."))

    # Compute total change
    change = utils.compute_percent_change(context, presc)

    return warnings, f"{change * 100:.2f}"


@app.callback(
    Output("presc-fig", "figure"),
    Input("chart-select", "value"),
    Input({"type": "presc-slider", "index": ALL}, "value"),
    State("year-input", "value"),
    State("lat-dropdown", "value"),
    State("lon-dropdown", "value"),
    prevent_initial_call=True
)
def update_presc_chart(chart_type, sliders, year, lat, lon):
    """
    Updates prescription pie from store according to chart type.
    :param chart_type: String input from chart select dropdown.
    :param sliders: Prescribed slider values.
    :param year: Selected context year (also for title of chart).
    :param lat: Selected context lat.
    :param lon: Selected context lon.
    :return: New chart of type chart_type using presc data.
    """

    # If we have no prescription just return an empty chart
    if all(slider == 0 for slider in sliders):
        return utils.create_treemap(pd.Series([]), type_context=False, year=year)

    presc = pd.Series(sliders, index=constants.RECO_COLS)
    context = df.loc[year, lat, lon]

    chart_data = context[constants.LAND_USE_COLS].copy()
    chart_data[constants.RECO_COLS] = presc[constants.RECO_COLS]

    # Manually calculate nonland from context so that it's not zeroed out by sliders.
    nonland = 1 - context[constants.LAND_USE_COLS].sum()
    nonland = nonland if nonland > 0 else 0
    chart_data["nonland"] = nonland

    assert chart_type in ("Treemap", "Pie Chart")

    if chart_type == "Treemap":
        return utils.create_treemap(chart_data, type_context=False, year=year)
    
    return utils.create_pie(chart_data, type_context=False, year=year)


@app.callback(
    Output({"type": "presc-slider", "index": ALL}, "value", allow_duplicate=True),
    Input("sum-button", "n_clicks"),
    State({"type": "presc-slider", "index": ALL}, "value"),
    State("year-input", "value"),
    State("lat-dropdown", "value"),
    State("lon-dropdown", "value"),
    State("locks", "value"),
    prevent_initial_call=True
)
def sum_to_1(n_clicks, sliders, year, lat, lon, locked):
    """
    Sets slider values to sum to how much land was used in context.
    Subtracts locked sum from both of these and doesn't adjust them.
    :param n_clicks: Unused number of times button has been clicked.
    :param sliders: Prescribed slider values to set to sum to 1.
    :param year: Selected context year.
    :param lat: Selected context lat.
    :param lon: Selected context lon.
    :param locked: Which sliders to not consider in calculation.
    :return: Slider values scaled down to fit percentage of land used in context.
    """
    context = df.loc[year, lat, lon]
    presc = pd.Series(sliders, index=constants.RECO_COLS)

    old_sum = context[constants.RECO_COLS].sum()
    new_sum = presc.sum()

    # TODO: There is certainly a more elegant way to handle this.
    if locked:
        unlocked = [col for col in constants.RECO_COLS if col not in locked]
        locked_sum = presc[locked].sum()
        old_sum -= locked_sum
        new_sum -= locked_sum
        # We do this to avoid divide by zero. In the case where new_sum == 0
        # we have all locked columns and/or zero columns so no adjustment is needed
        if new_sum != 0:
            presc[unlocked] = presc[unlocked].div(new_sum).mul(old_sum)

    else:
        presc = presc.div(new_sum).mul(old_sum)

    # Set all negative values to 0
    presc[presc < 0] = 0
    return presc.tolist()


@app.callback(
    Output("predict-eluc", "value"),
    Input("predict-button", "n_clicks"),
    State("year-input", "value"),
    State("lat-dropdown", "value"),
    State("lon-dropdown", "value"),
    State({"type": "presc-slider", "index": ALL}, "value"),
    State("pred-select", "value"),
    prevent_initial_call=True
)
def predict(n_clicks, year, lat, lon, sliders, predictor_name):
    """
    Predicts ELUC from context and prescription stores.
    :param n_clicks: Unused number of times button has been clicked.
    :param year: Selected context year.
    :param lat: Selected context lat.
    :param lon: Selected context lon.
    :param sliders: Prescribed slider values.
    :param predictor_name: String name of predictor to use from dropdown.
    :return: Predicted ELUC.
    """
    context = df.loc[year, lat, lon]
    presc = pd.Series(sliders, index=constants.RECO_COLS)

    # Preprocess presc into diffs
    presc = presc.combine_first(context[constants.NO_CHANGE_COLS])
    diff = presc[constants.LAND_USE_COLS] - context[constants.LAND_USE_COLS]
    diff = diff.rename(constants.COLS_MAP)
    diff_df = pd.DataFrame([diff])

    predictor = None
    prediction = 0
    if predictor_name == "Random Forest":
        predictor = random_forest_predictor
        prediction = predictor.predict(diff_df)
        return f"{prediction:.4f}"

    else:
        return "0"


@app.callback(
    Output("total-em", "children"),
    Output("tickets", "children"),
    Output("people", "children"),
    Input("predict-eluc", "value"),
    State("year-input", "value"),
    State("lat-dropdown", "value"),
    State("lon-dropdown", "value"),
    prevent_initial_call=True
)
def update_trivia(eluc_str, year, lat, lon):
    """
    Updates trivia section based on rounded ELUC value.
    :param eluc_str: ELUC in string form.
    :param year: Selected context year.
    :param lat: Selected context lat.
    :param lon: Selected context lon.
    :return: Trivia string output.
    """
    context = df.loc[year, lat, lon]
    area = context["cell_area"]

    # Calculate total reduction
    eluc = float(eluc_str)
    total_reduction = eluc * area
    return f"{-1 * total_reduction:,.2f} tonnes CO2", \
            f"{-1 * total_reduction // constants.CO2_JFK_GVA:,.0f} tickets", \
                f"{-1 * total_reduction // constants.CO2_PERSON:,.0f} people"


app.title = 'Land Use Optimization'
app.css.config.serve_locally = False
# Don't be afraid of the 3rd party URLs: chriddyp is the author of Dash!
# These two allow us to dim the screen while loading.
# See discussion with Dash devs here: https://community.plotly.com/t/dash-loading-states/5687
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/brPBPO.css'})

app.layout = html.Div([
    dcc.Markdown('''
# Land Use Optimization
This site is for demonstration purposes only.

For a given context cell representing a portion of the earth,
identified by its latitude and longitude coordinates, and a given year:
* What changes can we make to the land usage
* In order to minimize the resulting estimated CO2 emissions? (Emissions from Land Use Change, ELUC, 
in tons of carbon per hectare)
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
        dcc.Graph(id='context-fig', figure=utils.create_treemap(type_context=True), style={'grid-column': '3'}),
        dcc.Graph(id='presc-fig', figure=utils.create_treemap(type_context=False), style={'grid-clumn': '4'})
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


if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=8080)
