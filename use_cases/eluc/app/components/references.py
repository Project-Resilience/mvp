"""
Simple component that returns a div with references to the data, code, etc. used in the project.
"""
from dash import html


# pylint: disable=too-few-public-methods
class ReferencesComponent:
    """
    Component that returns a div with references to the data, code, etc. used in the project.
    """
    def get_references_div(self):
        """
        Creates div with various references. Uses the same styling as the trivia div.
        """
        inline_block = {"display": "inline-block", "padding-right": "10px"}
        references_div = html.Div([
            html.Div(className="parent", children=[
                html.P("Code for this project can be found here:  ",
                       className="child", style=inline_block),
                html.A("(Project Resilience MVP repo)",
                       href="https://github.com/Project-Resilience/mvp/tree/main/use_cases/eluc\n"),
            ]),
            html.Div(className="parent", children=[
                html.P("The paper for this project can be found here:  ",
                       className="child", style=inline_block),
                html.A("(arXiv link)", href="https://arxiv.org/abs/2311.12304\n"),
            ]),
            html.Div(className="parent", children=[
                html.P("ELUC data provided by the BLUE model  ",
                       className="child", style=inline_block),
                html.A("(BLUE: Bookkeeping of land use emissions)",
                       href="https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014GB004997\n"),
            ]),
            html.Div(className="parent", children=[
                html.P("Land use change data provided by the LUH2 project",
                       className="child", style=inline_block),
                html.A("(LUH2: Land Use Harmonization 2)", href="https://luh.umd.edu/\n"),
            ]),
            html.Div(className="parent", children=[
                html.P("Setup is described in Appendix C2.1 of the GCB 2022 report",
                       className="child", style=inline_block),
                html.A("(Global Carbon Budget 2022 report)",
                       href="https://essd.copernicus.org/articles/14/4811/2022/#section10/\n"),
            ]),
            html.Div(className="parent", children=[
                html.P("The Global Carbon Budget report assesses the global CO2 budget \
                    for the Intergovernmental Panel on Climate Change",
                       className="child", style=inline_block),
                html.A("(IPCC)", href="https://www.ipcc.ch/\n"),
            ]),
        ])
        return references_div
