"""
Component showing the little intro blurb.
"""
from dash import html


class IntroComponent():
    """
    Title card component
    """
    def get_div(self):
        """
        Creates the intro title card describing the project.
        """
        div = html.Div(
            className="mb-5 d-flex flex-column align-items-center justify-content-center",
            children=[
                html.H2("Land Use Optimization", className="text-center w-100 mb-5 mt-5"),
                html.P([
                    "Allocation of land for different uses significantly affects carbon balance and climate \
                    change. A surrogate model learned from historical land-use changes and carbon \
                    emission simulations allows efficient evaluation of such allocations. An evolutionary \
                    search then discovers effective land-use policies for specific locations. This \
                    system, using the technology behind ",
                    html.A("Cognizant NeuroAI", href="https://www.cognizant.com/us/en/services/ai/ai-lab"),
                    " and built on the ",
                    html.A("Project Resilience platform",
                           href="https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/Pages/project-resilience.aspx"),
                    ", generates Pareto fronts trading off carbon impact and amount of change customized to different \
                    locations, offering a useful tool for land-use planning."
                ], className="w-50"),
            ]
        )

        return div
