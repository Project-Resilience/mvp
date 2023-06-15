from dash import Dash, html, dash_table, dcc, Input, Output, State, MATCH, ALL
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import pandas as pd
from math import isclose

from constants import LAND_USE_COLS, CONTEXT_COLUMNS, PRESCRIPTOR_LIST, CHART_COLS, ALL_LAND_USE_COLS, SLIDER_PRECISION
from Prescriptor import Prescriptor
from Predictor import Predictor
from utils import add_nonland

app = Dash(__name__)

# TODO: should we load all our data into a store?
# This seems more secure.
df = pd.read_csv("../data/gcb/processed/uk_eluc.csv")

PIE_DATA = [0 for _ in range(len(CHART_COLS) - 1)]
PIE_DATA.append(1)

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])
fig.add_pie(values=PIE_DATA, labels=CHART_COLS, title="Initial", row=1, col=1)
fig.add_pie(values=PIE_DATA, labels=CHART_COLS, title="Prescribed", row=1, col=2)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

sliders = [
    dcc.Slider(
        min=0, 
        max=1, 
        step=SLIDER_PRECISION,
        value=0,
        marks=None,
        tooltip={"placement": "bottom", "always_visible": False}, 
        id={"type": "presc-slider", "index": f"{col}-slider"}) for col in LAND_USE_COLS]

locked_cols = [col for col in CHART_COLS if col not in LAND_USE_COLS]
locked_inputs = [
    dcc.Input(
        value=0,
        type="number",
        disabled=True,
        id={"type": "locked-input", "index": f"{col}-locked"}) for col in locked_cols
]


@app.callback(
    Output("pies", "extendData", allow_duplicate=True),
    Output("context-store", "data"),
    Output({"type": "locked-input", "index": ALL}, "value"),
    Input("context-button", "n_clicks"),
    State("lat-input", "value"),
    State("lon-input", "value"),
    State("time-input", "value"),
    prevent_initial_call=True
)
def select_context(n_clicks, lat, lon, time):
    context = df[(df['i_lat'] == lat) & (df['i_lon'] == lon) & (df['time'] == time)]
    chart_df = add_nonland(context[ALL_LAND_USE_COLS])
    chart_data = chart_df.iloc[0].tolist()
    new_data = [{
            "labels": [CHART_COLS],
            "values": [chart_data]},
        [0], len(CHART_COLS)
    ]
    return new_data, context.to_dict("records"), chart_df[locked_cols].iloc[0].tolist()

@app.callback(
    Output({"type": "presc-slider", "index": ALL}, "value", allow_duplicate=True),
    Input("presc-button", "n_clicks"),
    State("presc-dropdown", "value"),
    State("context-store", "data"),
    prevent_initial_call=True
)
def select_prescriptor(n_clicks, presc_id, context):
    # TODO: this is pretty lazy. We should cache used prescriptors
    prescriptor = Prescriptor(presc_id)
    context_df = pd.DataFrame.from_records(context)[CONTEXT_COLUMNS]
    prescribed = prescriptor.run_prescriptor(context_df)
    return prescribed[LAND_USE_COLS].iloc[0].tolist()

@app.callback(
    Output("presc-store", "data"),
    Input({"type": "presc-slider", "index": ALL}, "value"),
    prevent_initial_call=True
)
def store_prescription(sliders):
    presc = pd.DataFrame([sliders], columns=LAND_USE_COLS)
    return presc.to_dict("records")

@app.callback(
    Output("pies", "extendData", allow_duplicate=True),
    Input("presc-store", "data"),
    State("context-store", "data"),
    prevent_initial_call=True
)
def update_chart(presc, context):
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
    context_df = pd.DataFrame.from_records(context)[LAND_USE_COLS]
    presc_df = pd.DataFrame.from_records(presc)[LAND_USE_COLS]
    new_sum = presc_df.sum(axis=1)
    old_sum = context_df.sum(axis=1)
    return presc_df.div(new_sum, axis=0).mul(old_sum, axis=0).iloc[0].tolist()

@app.callback(
    Output("prediction", "children"),
    Input("predict-button", "n_clicks"),
    State("context-store", "data"),
    State("presc-store", "data"),
    prevent_initial_call=True
)
def predict(n_clicks, context, presc):
    context_df = pd.DataFrame.from_records(context)[CONTEXT_COLUMNS]
    presc_df = pd.DataFrame.from_records(presc)[LAND_USE_COLS]
    predictor = Predictor()
    prediction, change = predictor.run_predictor(context_df, presc_df)
    out = f"Prediction: {prediction} ELUC; Change: {change}"

    # Check if prescriptions sum to 1
    # TODO: Are we being precise enough?
    new_sum = presc_df.sum(axis=1).iloc[0]
    old_sum = context_df[LAND_USE_COLS].sum(axis=1).iloc[0]
    if not isclose(new_sum, old_sum, rel_tol=1e-7):
        out = "WARNING: prescriptions sum to " + str(new_sum) + " instead of " + str(old_sum) + " " + out
    
    return out


app.layout = html.Div([
    dcc.Store(id='context-store'),
    dcc.Store(id='presc-store'),
    html.Div(
        [dcc.Input(id='lat-input', type='number', value=51.625),
        dcc.Input(id='lon-input', type='number', value=-3.375),
        dcc.Input(id='time-input', type='number', value=2021),
        html.Button("Submit Context", id='context-button', n_clicks=0)
    ]),
    html.Div([
        dcc.Dropdown(id='presc-dropdown', options=PRESCRIPTOR_LIST, placeholder="Select a Prescriptor"),
        html.Button("Prescribe", id='presc-button', n_clicks=0)
    ]),
    dcc.Graph(id='pies', figure=fig),
    html.Div([
        html.Div(sliders),
        html.Div(locked_inputs),
        html.Button("Sum to 1", id='sum-button', n_clicks=0)
    ]),
    html.Div([
        html.Button("Predict", id='predict-button', n_clicks=0),
        html.Div(id='prediction')
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)