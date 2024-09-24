"""
Main app file for ELUC demo.
Uses many 'components' to separate divs and their related callbacks.
They aren't necessarily truly reusable components, but they help to organize the code.
"""
import pandas as pd
from dash import Dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

import app.constants as app_constants
from app.components.chart import ChartComponent
from app.components.legend import LegendComponent
from app.components.lock import LockComponent
from app.components.map import MapComponent
from app.components.prediction import PredictionComponent
from app.components.prescription import PrescriptionComponent
from app.components.references import ReferencesComponent
from app.components.sliders import SlidersComponent
from app.components.trivia import TriviaComponent

app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
           prevent_initial_callbacks="initial_duplicate")
server = app.server

df = pd.read_csv(app_constants.DATA_FILE_PATH, index_col=app_constants.INDEX_COLS)

legend_component = LegendComponent()

map_component = MapComponent(df)
map_component.register_update_map_callback(app)
map_component.register_click_map_callback(app)
map_component.register_select_country_callback(app)

prescription_component = PrescriptionComponent(df)
prescription_component.register_select_prescriptor_callback(app)
prescription_component.register_toggle_modal_callback(app)

sliders_component = SlidersComponent(df)
sliders_component.register_set_frozen_reset_sliders_callback(app)
sliders_component.register_show_slider_value_callback(app)
sliders_component.register_sum_to_one_callback(app)

lock_component = LockComponent()

chart_component = ChartComponent(df)
chart_component.register_update_context_chart_callback(app)
chart_component.register_update_presc_chart_callback(app)

prediction_component = PredictionComponent(df)
prediction_component.register_predictor_callback(app)
prediction_component.register_land_use_callback(app)

trivia_component = TriviaComponent(df)
trivia_component.register_update_trivia_callback(app)

references_component = ReferencesComponent()

app.title = 'Land Use Optimization'
app.css.config.serve_locally = False
# Don't be afraid of the 3rd party URLs: chriddyp is the author of Dash!
# These two allow us to dim the screen while loading.
# See discussion with Dash devs here: https://community.plotly.com/t/dash-loading-states/5687
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/brPBPO.css'})

app.layout = html.Div([
#     dcc.Markdown('''
# # Land Use Optimization
# This site is for demonstration purposes only.

# For a given context cell representing a portion of the earth,
# identified by its latitude and longitude coordinates, and a given year:
# * What changes can we make to the land usage
# * In order to minimize the resulting estimated CO2 emissions? (Emissions from Land Use Change, ELUC,
#  in tons of carbon per hectare)

# '''),
#     dcc.Markdown('''## Context'''),
#     html.Div([
#         dcc.Graph(id="map", figure=map_component.get_map_fig(), style={"grid-column": "1"}),
#         html.Div([map_component.get_context_div()], style={"grid-column": "2"}),
#         html.Div([legend_component.get_legend_div()], style={"grid-column": "3"})
#     ], style={"display": "grid", "grid-template-columns": "auto 1fr auto", 'position': 'relative'}),
    html.Div(
        children=[
            dbc.Row(
                html.H2("Land Use Optimization", className="display-4 w-50 mx-auto text-center mb-3")
            ),
            dbc.Row(
                html.P("This site is for demonstration purposes only. For a given context cell representing a portion \
                       of the earth, identified by its latitude and longitude coordinates, and a given year: \
                       what changes can we make to the land usage in order to minimize the resulting estimated CO2 \
                       emissions? (Emissions from Land Use Change, ELUC, in tonnes of carbon per hectare)",
                       className="lead w-50 mx-auto text-center")
            ),
            dbc.Row(
                style={"height": "80vh"}
            ),
            dbc.Row(
                html.P("Get Started:", className="w-50 text-center mx-auto text-white h4")
            ),
            dbc.Row(
                html.I(className="bi bi-arrow-up w-50 text-center mx-auto text-white h1")
            ),
            dbc.Row(
                style={"height": "5vh"}
            )
        ]
    ),
    html.Div(
        className="p-3 bg-white rounded-5 mx-auto w-75 mb-3",
        children=[
            dbc.Container(
                fluid=True,
                className="py-3 d-flex flex-column h-100 align-items-center",
                children=[
                    html.H2("Context", className="text-center mb-5"),
                    dbc.Row(
                        className="w-100",
                        children=[
                            dbc.Row(
                                children=[
                                    dbc.Col(dcc.Graph(id="map", figure=map_component.get_map_fig())),
                                    dbc.Col(map_component.get_context_div())
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    ),
    html.Div(
        className="p-3 bg-white rounded-5 mx-auto w-75 mb-3",
        children=[
            dbc.Container(
                fluid=True,
                className="py-3 d-flex flex-column h-100 align-items-center",
                children=[
                    html.H2("Actions", className="text-center mb-5"),
                    dbc.Row(
                        className="w-100",
                        children=[
                            dbc.Row(
                                children=[
                                    dbc.Col(prescription_component.get_presc_select_div()),
                                    dbc.Col(chart_component.get_chart_select_div())
                                ]
                            )
                        ]
                    ),
                    dbc.Row(
                        className="w-100",
                        children=[
                            dbc.Col(lock_component.get_checklist_div(), className="h-100"),
                            dbc.Col(sliders_component.get_sliders_div(), className="h-100"),
                            dbc.Col(dcc.Graph(id='context-fig', figure=chart_component.create_treemap(type_context=True))),
                            dbc.Col(dcc.Graph(id='presc-fig', figure=chart_component.create_treemap(type_context=False)))
                        ]
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                sliders_component.get_frozen_div(),
                                width=6
                            )
                        ]
                    ),
                    dbc.Row(
                        className="w-100",
                        children=[
                            dbc.Col(
                                children=[
                                    html.Button("Sum to 100%", id='sum-button', n_clicks=0),
                                    html.Div(id='sum-warning')
                                ],
                                width=6
                            )
                        ]
                    )
                ]
            )
        ]
    ),
    # dcc.Markdown('''## Actions'''),
    # html.Div([
    #     html.Div([prescription_component.get_presc_select_div()], style={"grid-column": "1"}),
    #     html.Div([chart_component.get_chart_select_div()],
    #              style={"grid-column": "2", "margin-top": "-10px", "margin-left": "10px"}),
    # ], style={"display": "grid", "grid-template-columns": "45% 15%"}),
    # html.Div([
    #     html.Div(lock_component.get_checklist_div(), style={"grid-column": "1", "height": "100%"}),
    #     html.Div(sliders_component.get_sliders_div(), style={'grid-column': '2'}),
    #     dcc.Graph(id='context-fig',
    #               figure=chart_component.create_treemap(type_context=True),
    #               style={'grid-column': '3'}),
    #     dcc.Graph(id='presc-fig',
    #               figure=chart_component.create_treemap(type_context=False),
    #               style={'grid-clumn': '4'})
    # ], style={'display': 'grid', 'grid-template-columns': '4.5% 40% 1fr 1fr', "width": "100%"}),
    # # The above line can't be set to auto because the lines will overflow!
    # html.Div([
    #     sliders_component.get_frozen_div(),
    #     html.Button("Sum to 100%", id='sum-button', n_clicks=0),
    #     html.Div(id='sum-warning')
    # ]),
    dcc.Markdown('''## Outcomes'''),
    prediction_component.get_predict_div(),
    dcc.Markdown('''## Trivia'''),
    trivia_component.get_trivia_div(),
    dcc.Markdown('''## References'''),
    references_component.get_references_div()
], style={'padding-left': '10px'},)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False, port=4057, use_reloader=True, threaded=False)
