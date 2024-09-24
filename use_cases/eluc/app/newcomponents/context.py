
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd

from app.components.map import MapComponent
from app.utils import add_nonland, EvolutionHandler
from data import constants

class ContextComponent():
    def __init__(self, app_df: pd.DataFrame, handler: EvolutionHandler):
        self.map_component = MapComponent(app_df)
        self.app_df = app_df
        self.handler = EvolutionHandler()

    def get_div(self):
        div = html.Div(
            className="w-50 vh-50",
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(
                            dcc.Graph(id="map", figure=self.map_component.get_map_fig())
                        ),
                        dbc.Col(
                            children=[
                                self.map_component.get_context_div(),
                                dbc.Button("Prescribe", id="presc-button")
                            ]
                        )
                    ]
                )
            ]
        )
        return div
    
    def register_callbacks(self, app):
        self.map_component.register_click_map_callback(app)
        self.map_component.register_select_country_callback(app)
        self.map_component.register_update_map_callback(app)

        @app.callback(
            Output("results-store", "data"),
            Input("presc-button", "n_clicks"),
            State("year-input", "value"),
            State("lat-dropdown", "value"),
            State("lon-dropdown", "value"),
            prevent_initial_call=True
        )
        def run_prescription(n_clicks: int, year: int, lat: float, lon: float) -> dict[str: list]:
            if n_clicks is not None and n_clicks >= 1:
                condition = (self.app_df["time"] == year) & (self.app_df["lat"] == lat) & (self.app_df["lon"] == lon)
                context_df = self.app_df[condition]
                context_df = context_df[constants.CAO_MAPPING["context"]].iloc[0:1]
                results_df = self.handler.prescribe_all(context_df)
                results_json = results_df.to_dict(orient="records")
                print(f"Updating store with {len(results_df)} prescriptions.")
                return results_json
