"""
File in charge of handling the sliders on the app.
"""
from dash import Input, State, Output, ALL, MATCH
from dash import dcc
from dash import html
import pandas as pd

from app import constants as app_constants
from app import utils
from data import constants

class SlidersComponent:
    """
    Component that displays the sliders, shows their values in frozen inputs, resets the sliders when context changes,
    and controls the sum to 1 button.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_sliders_div(self):
        """
        Returns div with sliders for each recommended land-use.
        Gets updated by the prescriptor.
        """
        sliders_div = html.Div([
            html.Div([
                html.Div([
                    dcc.Slider(
                        min=0,
                        max=1,
                        step=app_constants.SLIDER_PRECISION,
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
        return sliders_div

    def get_frozen_div(self):
        """
        Frozen input boxes that we use to display the slider values.
        """
        frozen_div = html.Div([
            dcc.Input(
                value=f"{col}: 0.00%",
                type="text",
                disabled=True,
                id={"type": "frozen-input", "index": f"{col}-frozen"})
                for col in app_constants.NO_CHANGE_COLS + ["nonland"]
        ])
        return frozen_div

    def register_set_frozen_reset_sliders_callback(self, app):
        """
        Registers function that resets sliders to 0 when context changes.
        """
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
            context = self.df.loc[year, lat, lon]

            chart_data = utils.add_nonland(context[constants.LAND_USE_COLS])

            frozen_cols = app_constants.NO_CHANGE_COLS + ["nonland"]
            frozen = chart_data[frozen_cols].tolist()
            frozen = [f"{frozen_cols[i]}: {frozen[i]*100:.2f}%" for i in range(len(frozen_cols))]

            reset = [0 for _ in constants.RECO_COLS]

            max_val = chart_data[constants.RECO_COLS].sum()
            maxes = [max_val for _ in range(len(constants.RECO_COLS))]

            return frozen, reset, maxes

    def register_show_slider_value_callback(self, app):
        """
        Registers the callback that shows the slider values next to the sliders.
        """
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

    def register_sum_to_one_callback(self, app):
        """
        Registers callback that makes it so that when you click the sum to 1 button it sums the sliders to 1.
        """
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
        def sum_to_1(_, sliders, year, lat, lon, locked):
            """
            Sets slider values to sum to how much land was used in context.
            Subtracts locked sum from both of these and doesn't adjust them.
            :param sliders: Prescribed slider values to set to sum to 1.
            :param year: Selected context year.
            :param lat: Selected context lat.
            :param lon: Selected context lon.
            :param locked: Which sliders to not consider in calculation.
            :return: Slider values scaled down to fit percentage of land used in context.
            """
            context = self.df.loc[year, lat, lon]
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
