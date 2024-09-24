"""
This file is a component that handles the map where the user can select context and its associated callbacks.
"""
from dash import Input, State, Output, dcc, html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import regionmask

from app import constants as app_constants


class MapComponent:
    """
    Component handling the map. Keeps track of the latitudes and longitudes a user can select as well as the countries.
    """
    # pylint: disable=too-many-instance-attributes
    # These instance attributes make the code much more readable
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Cells
        self.min_lat = df["lat"].min()
        self.max_lat = df["lat"].max()
        self.min_lon = df["lon"].min()
        self.max_lon = df["lon"].max()
        self.min_time = df["time"].min()
        self.max_time = df["time"].max()

        self.lat_list = list(np.arange(self.min_lat, self.max_lat + app_constants.GRID_STEP, app_constants.GRID_STEP))
        self.lon_list = list(np.arange(self.min_lon, self.max_lon + app_constants.GRID_STEP, app_constants.GRID_STEP))

        self.countries_df = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.to_dataframe()

        self.map_fig = go.Figure()

    def get_map_fig(self):
        """
        Returns the map figure that gets updated by the callbacks.
        """
        return self.map_fig

    def get_context_div(self):
        """
        Div that allows the user to select context inputs manually instead of from the map.
        """
        context_div = html.Div(
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(html.P("Region")),
                        dbc.Col(
                            dcc.Dropdown(
                                id="loc-dropdown",
                                options=list(self.countries_df["names"]),
                                value=list(self.countries_df["names"])[143]
                            )
                        )
                    ]
                ),
                dbc.Row(
                    children=[
                        dbc.Col(html.P("Latitude")),
                        dbc.Col(
                            dcc.Dropdown(
                                id="lat-dropdown",
                                options=[{"label": lat, "value": lat} for lat in self.lat_list],
                                value=51.625,
                            )
                        )
                    ]
                ),
                dbc.Row(
                    children=[
                        dbc.Col(html.P("Longitude")),
                        dbc.Col(
                            dcc.Dropdown(
                                id="lon-dropdown",
                                options=[{"label": lon, "value": lon} for lon in self.lon_list],
                                value=-3.375,
                            )
                        )
                    ]
                ),
                dbc.Row(
                    children=[
                        dbc.Col(html.P("Year")),
                        dbc.Col(
                            html.Div([
                                dcc.Input(
                                    id="year-input",
                                    type="number",
                                    value=2021,
                                    debounce=True
                                ),
                                dcc.Tooltip(f"Year must be between {self.min_time} and {self.max_time}.")
                            ])
                            
                        )
                    ]
                )
            ]
        )
        return context_div

    def register_select_country_callback(self, app):
        """
        Callback in charge of changing the selected country and relocating the map to a valid lat/lon.
        We just take the middle sample of the country's data.
        """
        @app.callback(
            Output("lat-dropdown", "value", allow_duplicate=True),
            Output("lon-dropdown", "value", allow_duplicate=True),
            Input("loc-dropdown", "value"),
            State("year-input", "value"),
            prevent_initial_call=True
        )
        def select_country(location, year):
            """
            Changes the selected country and relocates map to a valid lat/lon.
            This makes the update_map function only load the current country's data.
            :param location: Selected country name.
            :param year: Used to get proper # of points to sample from.
            :return: A sample latitude/longitude point within the selected country.
            """
            country_idx = self.countries_df[self.countries_df["names"] == location].index[0]
            samples = self.df[self.df["country"] == country_idx].loc[year]
            example = samples.iloc[len(samples) // 2]
            return example.name[0], example.name[1]

    def register_click_map_callback(self, app):
        """
        Registers callback which updates the lat and lon dropdowns when the map is clicked.
        """
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

    def register_update_map_callback(self, app):
        """
        Registers callback that updates the map when year, lat, or lon is changed.
        """
        @app.callback(
            Output("map", "figure"),
            Input("year-input", "value"),
            Input("lat-dropdown", "value"),
            Input("lon-dropdown", "value"),
            State("loc-dropdown", "value"),
        )
        def update_map(year, lat, lon, location):
            """
            Updates map data behind the scenes when year is clicked.
            Changes focus when region is selected.
            :param location: Selected country name.
            :param year: The selected year.
            :return: A newly created map.
            """
            country_idx = self.countries_df[self.countries_df["names"] == location].index[0]
            # Filter data by year and location
            data = self.df.loc[year]
            data = data[data["country"] == country_idx]
            # Drop index because plotly requires single integer index
            data = data.copy().reset_index(drop=True)

            # Find colored point
            lat_lon = (data["lat"] == lat) & (data["lon"] == lon)
            idx = data[lat_lon].index[0]

            return self.create_map(data, 10, idx)

    def create_map(self, df: pd.DataFrame, zoom=10, color_idx=None) -> go.Figure:
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
