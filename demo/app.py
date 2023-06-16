from math import isclose, log10

import numpy as np
import pandas as pd
from dash import ALL
from dash import MATCH
from dash import Dash
from dash import Input
from dash import Output
from dash import State
from dash import dcc
from dash import html
from plotly.subplots import make_subplots

from Predictor import Predictor
from Prescriptor import Prescriptor
from constants import ALL_LAND_USE_COLS
from constants import CHART_COLS
from constants import CONTEXT_COLUMNS
from constants import LAND_USE_COLS
from constants import PRESCRIPTOR_LIST
from constants import SLIDER_PRECISION
from utils import add_nonland
from utils import round_list


app = Dash(__name__)

# TODO: should we load all our data into a store?
# This seems more secure.
df = pd.read_csv("../data/gcb/processed/uk_eluc.csv")

# Cells
GRID_STEP = 0.25
min_lat = df["i_lat"].min()
max_lat = df["i_lat"].max()
min_lon = df["i_lon"].min()
max_lon = df["i_lon"].max()
min_year = df["time"].min()
max_year = df["time"].max()

lat_list = [lat for lat in np.arange(min_lat, max_lat + GRID_STEP, GRID_STEP)]
lon_list = [lon for lon in np.arange(min_lon, max_lon + GRID_STEP, GRID_STEP)]

PIE_DATA = [0 for _ in range(len(CHART_COLS) - 1)]
PIE_DATA.append(1)

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])
fig.add_pie(values=PIE_DATA, labels=CHART_COLS, title="Initial", row=1, col=1)
fig.add_pie(values=PIE_DATA, labels=CHART_COLS, title="Prescribed", row=1, col=2)
fig.update_traces(textposition='inside')
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

context_div = html.Div([
                dcc.Markdown('''## Context'''),
                html.Div(
                    style={'display': 'flex'},
                    children=[
                        html.P("Lat", style={"display": "table-cell"}),
                        dcc.Dropdown(id='lat-dropdown',
                                     options=lat_list,
                                     placeholder="Select a latitude",
                                     value=51.625,
                                     style=dict(width='75%')
                                     ),
                        html.P("Lon", style={"display": "table-cell"}),
                        dcc.Dropdown(id='lon-dropdown',
                                     options=lon_list,
                                     placeholder="Select a longitude",
                                     value=-3.375,
                                     style=dict(width='75%')),
                        html.P("Year ", style={"display": "table-cell"}),
                        dcc.Input(id="year-input",
                                  type="number",
                                  value=2021,
                                  style={"display": "table-cell"}),
                        dcc.Tooltip(f"Year must be between {min_year} and {max_year}."),
                        html.Button("Submit Context", id='context-button', n_clicks=0)
                    ])
            ])

presc_select_div = html.Div([
    html.P("Minimize change", style={"grid-column": "1"}),
    html.Div([
        dcc.Slider(id='presc-select',
                min=0, max=len(PRESCRIPTOR_LIST)-1, step=1, 
                value=len(PRESCRIPTOR_LIST)//2,
                included=False,
                marks={i : "" for i in range(len(PRESCRIPTOR_LIST))})
    ], style={"grid-column": "2", "width": "100%", "margin-top": "8px"}),
    html.P("Minimize ELUC", style={"grid-column": "3", "padding-right": "10px"}),
    html.Button("Prescribe", id='presc-button', n_clicks=0, style={"grid-column": "4", "margin-top": "-10px"})
], style={"display": "grid", "grid-template-columns": "auto 1fr auto auto", "width": "45%", "align-content": "center"})

sliders_div = html.Div([
    html.Div([
        html.P(col, style={"grid-column": "1"}),
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
        ], style={"grid-column": "2", "width": "100%", "margin-top": "8px"}),
        html.Div("0", id={"type": "slider-value", "index": f"{col}-value"}, style={"grid-column": "3"}),
    ], style={"display": "grid", "grid-template-columns": "8% 1fr 20%"}) for col in LAND_USE_COLS]
)

locked_cols = [col for col in CHART_COLS if col not in LAND_USE_COLS]
locked_div = html.Div([
    dcc.Input(
        value=f"{col}: 0",
        type="text",
        disabled=True,
        id={"type": "locked-input", "index": f"{col}-locked"}) for col in locked_cols
])

predict_div = html.Div([
    html.Button("Predict", id='predict-button', n_clicks=0, style={"grid-column": "1"}),
    html.P("Predicted ELUC: ", style={"grid-column": "2"}),
    dcc.Input(
        type="number",
        disabled=True,
        id="predict-eluc",
        style={"grid-column": "3"}
    ),
    html.P("Land Change: ", style={"grid-column": "4"}),
    dcc.Input(
        type="number",
        disabled=True,
        id="predict-change",
        style={"grid-column": "5"}
    ),
], style={"display": "grid", "grid-template-columns": "auto auto 1fr auto 1fr"})



@app.callback(
    Output("pies", "extendData", allow_duplicate=True),
    Output("context-store", "data"),
    Output({"type": "locked-input", "index": ALL}, "value"),
    Input("context-button", "n_clicks"),
    State("lat-dropdown", "value"),
    State("lon-dropdown", "value"),
    State("year-input", "value"),
    prevent_initial_call=True
)
def select_context(n_clicks, lat, lon, year):
    """
    Loads context in from lon/lat/time. Updates pie chart, context data store, and locked inputs.
    :param n_clicks: Unused number of times button has been clicked.
    :param lat: Latitude to search.
    :param lon: Longitude to search.
    :param year: Year to search.
    :return: Updated pie data, context data to store, and locked slider values.
    """
    context = df[(df['i_lat'] == lat) & (df['i_lon'] == lon) & (df['time'] == year)]
    # For testing purposes:
    # context["primf"].iloc[0] = 0.2
    # context["primn"].iloc[0] = 0.1
    # context["pastr"].iloc[0] -= 0.3
    # context["secdf"].iloc[0] -= 0.1
    chart_df = add_nonland(context[ALL_LAND_USE_COLS])
    chart_data = chart_df.iloc[0].tolist()
    new_data = [{
            "labels": [CHART_COLS],
            "values": [chart_data]},
        [0], len(CHART_COLS)
    ]
    locked = chart_df[locked_cols].iloc[0].tolist()
    locked = round_list(locked)
    locked = [f"{locked_cols[i]}: {locked[i]}" for i in range(len(locked_cols))]

    return new_data, context.to_dict("records"), locked


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
    :param presc_id: Prescriptor id to load.
    :param context: Context data from store to run prescriptor on.
    :return: Updated slider values.
    """
    # TODO: this is pretty lazy. We should cache used prescriptors
    presc_id = PRESCRIPTOR_LIST[presc_idx]
    prescriptor = Prescriptor(presc_id)
    context_df = pd.DataFrame.from_records(context)[CONTEXT_COLUMNS]
    prescribed = prescriptor.run_prescriptor(context_df)
    return prescribed[LAND_USE_COLS].iloc[0].tolist()


@app.callback(
    Output("presc-store", "data"),
    Output({"type": "slider-value", "index": ALL}, "children"),
    Output("sum-warning", "children"),
    Input({"type": "presc-slider", "index": ALL}, "value"),
    State("context-store", "data"),
    prevent_initial_call=True
)
def store_prescription(sliders, context):
    """
    Stores slider values in store and displays them next to sliders.
    Warns user if values don't sum to 1.
    :param sliders: Slider values to store.
    :return: Stored slider values, slider values to display, warning if necessary.
    """
    context_df = pd.DataFrame.from_records(context)[CONTEXT_COLUMNS]
    presc = pd.DataFrame([sliders], columns=LAND_USE_COLS)
    rounded = round_list(presc.iloc[0].tolist())

    warning = ""
    # Check if prescriptions sum to 1
    # TODO: Are we being precise enough?
    new_sum = presc.sum(axis=1).iloc[0]
    old_sum = context_df[LAND_USE_COLS].sum(axis=1).iloc[0]
    if not isclose(new_sum, old_sum, rel_tol=1e-7):
        warning = "WARNING: prescriptions sum to " + str(new_sum) + " instead of " + str(old_sum)

    return presc.to_dict("records"), rounded, warning


@app.callback(
    Output("pies", "extendData", allow_duplicate=True),
    Input("presc-store", "data"),
    State("context-store", "data"),
    prevent_initial_call=True
)
def update_chart(presc, context):
    """
    Updates prescription pie from store.
    :param presc: Prescription data from store.
    :param context: Context data from store.
    :return: Updated prescription pie data.
    """
    presc_df = pd.DataFrame.from_records(presc)[LAND_USE_COLS]
    context_df = pd.DataFrame.from_records(context)[CONTEXT_COLUMNS]
    presc_df["primf"] = context_df["primf"]
    presc_df["primn"] = context_df["primn"]
    chart_data = add_nonland(presc_df[ALL_LAND_USE_COLS]).iloc[0].tolist()
    return [{
            "labels": [CHART_COLS],
            "values": [chart_data]},
        [1], len(CHART_COLS)
    ]


@app.callback(
    Output({"type": "presc-slider", "index": ALL}, "value", allow_duplicate=True),
    Input("sum-button", "n_clicks"),
    State("presc-store", "data"),
    State("context-store", "data"),
    prevent_initial_call=True
)
def sum_to_1(n_clicks, presc, context):
    """
    Sets slider values to sum to how much land was used in context.
    :param n_clicks: Unused number of times button has been clicked.
    :param presc: Prescription data from store.
    :param context: Context data from store.
    :return: Slider values scaled down to fit percentage of land used in context.
    """
    context_df = pd.DataFrame.from_records(context)[LAND_USE_COLS]
    presc_df = pd.DataFrame.from_records(presc)[LAND_USE_COLS]
    new_sum = presc_df.sum(axis=1)
    old_sum = context_df.sum(axis=1)
    return presc_df.div(new_sum, axis=0).mul(old_sum, axis=0).iloc[0].tolist()


@app.callback(
    Output("predict-eluc", "value"),
    Output("predict-change", "value"),
    Input("predict-button", "n_clicks"),
    State("context-store", "data"),
    State("presc-store", "data"),
    prevent_initial_call=True
)
def predict(n_clicks, context, presc):
    """
    Predicts ELUC from context and prescription stores.
    :param n_clicks: Unused number of times button has been clicked.
    :param context: Context data from store.
    :param presc: Prescription data from store.
    :return: Predicted ELUC and percent change.
    """
    context_df = pd.DataFrame.from_records(context)[CONTEXT_COLUMNS]
    presc_df = pd.DataFrame.from_records(presc)[LAND_USE_COLS]
    predictor = Predictor()
    prediction, change = predictor.run_predictor(context_df, presc_df)
    
    return prediction, change


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
        dcc.Store(id='presc-store'),
        dcc.Markdown('''
# Land Use Optimization
This site is for demonstration purposes only.

For a given context cell representing a portion of the earth,
identified by its latitude and longitude coordinates:
* what changes can we make to the land use
* in order to minimize the resulting estimated CO2 emissions (ELUC)?
'''),
        context_div,
        dcc.Markdown('''## Actions'''),
        presc_select_div,
        html.Div([
            html.Div(sliders_div, style={'grid-column': '1'}),
            dcc.Graph(id='pies', figure=fig, style={'grid-column': '2'})
        ], style={'display': 'grid', 'grid-template-columns': '40% 1fr'}),
        
        html.Div([
            #sliders_div,
            locked_div,
            html.Button("Sum to 1", id='sum-button', n_clicks=0),
            html.Div(id='sum-warning')
        ]),
        dcc.Markdown('''## Outcomes'''),
        predict_div
    ])

    app.run_server(debug=True)


if __name__ == '__main__':
    main()
