"""
Trivia component for the ELUC app showing how much carbon emissions are reduce in real-world terms.
"""
from dash import Input, State, Output
from dash import html
import pandas as pd

from app import constants as app_constants

class TriviaComponent():
    """
    Component in charge of generating the trivia div as well as updating it after prediction is made.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_trivia_div(self):
        """
        Returns a div with the same style as the references.
        Shows total emissions reduced, flight emissions, plane tickets mitigated, yearly carbon emissions of
        average world citizen, and number of peoples' carbon emissions mitigated from this change.
        """
        inline_block = {"display": "inline-block", "padding-right": "10px"}
        trivia_div = html.Div([
            html.Div(className="parent", children=[
                html.P(
                    "Total emissions reduced from this land use change: ",
                    className="child",
                    style=inline_block
                ),
                html.P(id="total-em", style={"font-weight": "bold"}|inline_block)
            ]),
            html.Div(className="parent", children=[
                html.I(className="bi bi-airplane", style=inline_block),
                html.P(
                    "Flight emissions from flying JFK to Geneva: ",
                    className="child",
                    style=inline_block
                ),
                html.P(f"{app_constants.CO2_JFK_GVA} tonnes CO2", style={"font-weight": "bold"}|inline_block)
            ]),
            html.Div(className="parent", children=[
                html.I(className="bi bi-airplane", style=inline_block),
                html.P(
                    "Plane tickets mitigated: ",
                    className="child",
                    style=inline_block
                ),
                html.P(id="tickets", style={"font-weight": "bold"}|inline_block)
            ]),
            html.Div(className="parent", children=[
                html.I(className="bi bi-person", style=inline_block),
                html.P(
                    "Total yearly carbon emissions of average world citizen: ",
                    className="child",
                    style=inline_block
                ),
                html.P(f"{app_constants.CO2_PERSON} tonnes CO2", style={"font-weight": "bold"}|inline_block)
            ]),
            html.Div(className="parent", children=[
                html.I(className="bi bi-person", style=inline_block),
                html.P(
                    "Number of peoples' carbon emissions mitigated from this change : ",
                    className="child",
                    style=inline_block
                ),
                html.P(id="people", style={"font-weight": "bold"}|inline_block)
            ]),
            html.P(
                "(Sources: https://flightfree.org/flight-emissions-calculator \
                    https://scied.ucar.edu/learning-zone/climate-solutions/carbon-footprint)",
                style={"font-size": "10px"}
            )
        ])
        return trivia_div

    def register_update_trivia_callback(self, app):
        """
        Registers callback that updates the trivia section when prediction is done.
        """
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
            context = self.df.loc[year, lat, lon]
            area = context["cell_area"]

            # Calculate total reduction
            eluc = float(eluc_str)
            total_reduction = eluc * area * app_constants.TC_TO_TCO2
            return f"{-1 * total_reduction:,.2f} tonnes CO2", \
                    f"{-1 * total_reduction // app_constants.CO2_JFK_GVA:,.0f} tickets", \
                        f"{-1 * total_reduction // app_constants.CO2_PERSON:,.0f} people"
