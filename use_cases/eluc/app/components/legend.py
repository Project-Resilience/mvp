"""
Simple component that returns a div with a markdown text that explains the land use types.
"""
from dash import dcc
from dash import html


# pylint: disable=too-few-public-methods
class LegendComponent:
    """
    Component with just a div that explains the land use types.
    """
    def get_legend_div(self):
        """
        Div explaining different land use types. Crop is now just one category.
        """
        # Legend examples come from https://hess.copernicus.org/preprints/hess-2021-247/hess-2021-247-ATC3.pdf
        legend_div = html.Div(
            style={"margin-bottom": "100px"},  # Because we removed some crops, we extend this so the map doesn't shrink
            children=[
                dcc.Markdown('''
        ### Land Use Types

        Primary: Vegetation that is untouched by humans

            - primf: Primary forest
            - primn: Primary nonforest vegetation


        Secondary: Vegetation that has been touched by humans

            - secdf: Secondary forest
            - secdn: Secondary nonforest vegetation

        Urban

        Crop

        Pasture

            - pastr: Managed pasture land
            - range: Natural grassland/savannah/desert/etc.
            ''')
            ]
        )
        return legend_div
