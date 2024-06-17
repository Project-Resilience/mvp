"""
File in charge of the lock component next to the sliders.
"""
from dash import dcc
from dash import html

from data import constants

class LockComponent:
    """
    Component that creates lock div based on reco columns.
    """
    def create_check_options(self, values: list) -> list:
        """
        Creates dash HTML options for checklist based on values.
        :param values: List of values to create options for.
        :return: List of dash HTML options.
        """
        options = []
        for val in values:
            options.append(
                {"label": [html.I(className="bi bi-lock"), html.Span(val)],
                "value": val})
        return options

    def get_checklist_div(self):
        """
        Creates check options based off reco cols then creates checklist div of locks.
        """
        check_options = self.create_check_options(constants.RECO_COLS)
        checklist_div = html.Div([
            dcc.Checklist(check_options, id="locks", inputStyle={"margin-bottom": "30px"})
        ])
        return checklist_div
