"""
Component showing the little intro blurb and the arrow to get started.
"""
from dash import html
import dash_bootstrap_components as dbc


class IntroComponent():
    """
    Title card component
    """
    def get_div(self):
        """
        Creates the intro title card describing the project.
        """
        div = html.Div(
            children=[
                dbc.Row(
                    html.H1("Land-Use Planning",
                            className="display-4 w-50 mx-auto text-center mb-3")
                ),
                dbc.Row(
                    html.P(
                        [
                            "Immediate action is required to combat climate change. The technology behind ",
                            html.A(
                                "Cognizant NeuroAI",
                                href="https://www.cognizant.com/us/en/services/ai/ai-lab",
                                style={"color": "black"}
                            ),
                            " brings automatic decision-making to the ",
                            html.A(
                                "En-ROADS platform",
                                href="https://www.climateinteractive.org/en-roads/",
                                style={"color": "black"}
                            ),
                            ", a powerful climate change simulator. A decision-maker can be ready for any \
                            scenario: choosing an automatically generated policy that suits their needs best, with the \
                            ability to manually modify the policy and see its results. This tool is brought together \
                            under ",
                            html.A(
                                "Project Resilience",
                                href="https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/\
                                    Pages/project-resilience.aspx",
                                style={"color": "black"}
                            ),
                            ", a United Nations initiative to use AI for good."
                        ],
                        className="lead w-50 mx-auto text-center"
                    )
                ),
                dbc.Row(
                    style={"height": "60vh"}
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
        )

        return div
