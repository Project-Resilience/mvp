"""
Context component class for selecting context.
"""
from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import regionmask

from app.components.context.map import MapComponent
from app.utils import EvolutionHandler
from data import constants


class ContextComponent():
    """
    Component containing map as well as dropdowns and input fields for picking a more specific context.
    """
    def __init__(self, app_df: pd.DataFrame, handler: EvolutionHandler):
        self.map_component = MapComponent(app_df)
        self.app_df = app_df
        self.handler = handler
        self.countries_df = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.to_dataframe()

    def create_label_and_value(self, label: str, value: html.Div) -> html.Div:
        """
        Standard dash function that pairs a label with any arbitrary value Div.
        """
        div = html.Div(
            className="d-flex flex-row",
            children=[
                html.Label(label, className="w-25"),
                html.Div(
                    value,
                    className="flex-grow-1"
                )
            ]
        )
        return div

    def get_div(self):
        """
        Returns the entire context div to display in the app.
        """
        div = html.Div(
            className="mb-5 mx-5",
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(
                            width={"offset": 3, "size": 3},
                            children=[
                                dcc.Graph(id="map", figure=self.map_component.get_map_fig()),
                                dcc.Dropdown(
                                    id="loc-dropdown",
                                    options=list(self.map_component.countries_df["names"]),
                                    value=list(self.map_component.countries_df["names"])[143]
                                )
                            ]
                        ),
                        dbc.Col(
                            width=3,
                            children=[
                                html.B("1. Select a land area on the map to optimize or manually enter coordinates."),
                                self.create_label_and_value(
                                    "Latitude",
                                    dcc.Dropdown(
                                        id="lat-dropdown",
                                        options=[{"label": lat,
                                                  "value": lat} for lat in self.map_component.lat_list],
                                        value=51.625,
                                    )
                                ),
                                self.create_label_and_value(
                                    "Longitude",
                                    dcc.Dropdown(
                                            id="lon-dropdown",
                                            options=[{"label": lon,
                                                      "value": lon} for lon in self.map_component.lon_list],
                                            value=-3.375,
                                    )
                                ),
                                self.create_label_and_value(
                                    "Year",
                                    html.Div([
                                            dcc.Input(
                                                id="year-input",
                                                type="number",
                                                value=2021,
                                                debounce=True
                                            ),
                                            dcc.Tooltip(f"Year must be between \
                                                        {self.map_component.min_time} and \
                                                            {self.map_component.max_time}.")
                                        ])
                                )
                            ]
                        )
                    ]
                )
            ]
        )
        return div

    def register_callbacks(self, app):
        """
        Registers callbacks to make app interactive. Registers old map callbacks as well as new one to run prescription.
        """
        self.map_component.register_click_map_callback(app)
        self.map_component.register_select_country_callback(app)
        self.map_component.register_update_map_callback(app)

        @app.callback(
            Output("results-store", "data"),
            Input("year-input", "value"),
            Input("lat-dropdown", "value"),
            Input("lon-dropdown", "value")
        )
        def run_prescription(year: int, lat: float, lon: float) -> dict[str: list]:
            """
            Runs prescription for the selected context on all prescriptors. Returns the results as a json to a store.
            """
            condition = (self.app_df["time"] == year) & (self.app_df["lat"] == lat) & (self.app_df["lon"] == lon)
            context_df = self.app_df[condition]
            context_df = context_df[constants.CAO_MAPPING["context"]].iloc[0:1]
            results_df = self.handler.prescribe_all(context_df)
            results_json = results_df.to_dict(orient="records")
            return results_json
