"""
File handling the visualization charts for the ELUC app.
"""
from dash import Input, State, Output, ALL
from dash import dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app import constants as app_constants
from app import utils
from data import constants


class ChartComponent:
    """
    Component in charge of handling the context and prescription charts.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_chart_select_div(self):
        """
        Div that allows the user to select between treemap and pie chart.
        """
        chart_select_div = dcc.Dropdown(
            options=app_constants.CHART_TYPES,
            id="chart-select",
            value=app_constants.CHART_TYPES[0],
            clearable=False
        )
        return chart_select_div

    def register_update_context_chart_callback(self, app):
        """
        Callback that updates the context chart when the context store is updated or the chart type is changed.
        """
        @app.callback(
            Output("context-fig", "figure"),
            # Input("chart-select", "value"),
            Input("year-input", "value"),
            Input("lat-dropdown", "value"),
            Input("lon-dropdown", "value")
        )
        # def update_context_chart(chart_type, year, lat, lon):
        def update_context_chart(year, lat, lon):
            """
            Updates context chart when context selection is updated or chart type is changed.
            :param chart_type: String input from chart select dropdown.
            :param year: Selected context year.
            :param lat: Selected context lat.
            :param lon: Selected context lon.
            :return: New figure type selected by chart_type with data context.
            """
            context = self.df.loc[[year], [lat], [lon]]
            chart_data = utils.add_nonland(context[constants.LAND_USE_COLS]).iloc[0]

            # assert chart_type in ("Treemap", "Pie Chart")

            # if chart_type == "Treemap":
            return self.create_treemap(chart_data, type_context=True, year=year)

            # return self.create_pie(chart_data.iloc[0], type_context=True, year=year)

    def register_update_presc_chart_callback(self, app):
        """
        Callback that updates prescription chart when prescription sliders are updated or chart type is changed.
        """
        @app.callback(
            Output("presc-fig", "figure"),
            Output("alert", "is_open"),
            Input("update-button", "n_clicks"),
            [State({"type": "diff-slider", "index": ALL}, "value")],
            State("year-input", "value"),
            State("lat-dropdown", "value"),
            State("lon-dropdown", "value"),
        )
        # def update_presc_chart(chart_type, sliders, year, lat, lon):
        def update_presc_chart(n_clicks, sliders, year, lat, lon):
            """
            Updates prescription chart from sliders according to chart type.
            :param chart_type: String input from chart select dropdown.
            :param sliders: Prescribed slider values.
            :param year: Selected context year (also for title of chart).
            :param lat: Selected context lat.
            :param lon: Selected context lon.
            :return: New chart of type chart_type using presc data.
            """

            # If we have no prescription just show the context chart
            bad_sliders = sum(sliders) < -0.01 or sum(sliders) > 0.01
            if all(slider == 0 for slider in sliders) or bad_sliders:
                context = self.df.loc[[year], [lat], [lon]]
                chart_data = utils.add_nonland(context[constants.LAND_USE_COLS]).iloc[0]
                return self.create_treemap(pd.Series(chart_data, dtype=float),
                                           type_context=False,
                                           year=year), bad_sliders

            diff = pd.Series(sliders, index=constants.RECO_COLS)
            context = self.df.loc[year, lat, lon]

            presc = context[constants.RECO_COLS] + diff

            chart_data = context[constants.LAND_USE_COLS].copy()
            chart_data[constants.RECO_COLS] = presc[constants.RECO_COLS]

            # Manually calculate nonland from context so that it's not zeroed out by sliders.
            nonland = 1 - context[constants.LAND_USE_COLS].sum()
            nonland = nonland if nonland > 0 else 0
            chart_data["nonland"] = nonland

            # if chart_type == "Treemap":
            return self.create_treemap(chart_data, type_context=False, year=year), bad_sliders
            # if chart_type == "Pie Chart":
                # return self.create_pie(chart_data, type_context=False, year=year)
            # raise ValueError(f"Invalid chart type: {chart_type}")

    def _create_hovertext(self, labels: list, parents: list, values: list, title: str) -> list:
        """
        Helper function that formats the hover text for the treemap to be 2 decimals.
        :param labels: Labels according to treemap format.
        :param parents: Parents for each label according to treemap format.
        :param values: Values for each label according to treemap format.
        :param title: Title of treemap, root node's name.
        :return: List of hover text strings.
        """
        hovertext = []
        for i, label in enumerate(labels):
            val = values[i] * 100
            # Get value of parent or 100 if parent is '' or 0
            if parents[i] == '' or values[labels.index(parents[i])] == 0:
                parent_v = values[0] * 100
            else:
                parent_v = values[labels.index(parents[i])] * 100
            if parents[i] == '':
                hovertext.append(f"{label}: {val:.2f}%")
            elif parents[i] == title:
                hovertext.append(f"{label}<br>{val:.2f}% of {title}")
            else:
                hovertext.append(f"{label}<br>{val:.2f}% of {title}<br>{(val/parent_v)*100:.2f}% of {parents[i]}")

        return hovertext

    def create_treemap(self, data=pd.Series, type_context=True, year=2021) -> go.Figure:
        """
        Creates a treemap figure from the given data.
        :param data: Pandas series of land use data
        :param type_context: If the title should be context or prescribed
        :return: Treemap figure
        """
        title = "Before" if type_context else "After"

        tree_params = {
            "branchvalues": "total",
            "sort": False,
            "texttemplate": "%{label}<br>%{percentRoot:.2%}",
            "hoverinfo": "label+percent root+percent parent",
            "root_color": "lightgrey"
        }

        labels, parents, values = None, None, None

        if data.empty:
            labels = [title]
            parents = [""]
            values = [1]

        else:
            total = data[constants.LAND_USE_COLS].sum()
            primary = data[app_constants.PRIMARY].sum()
            secondary = data[app_constants.SECONDARY].sum()
            fields = data[app_constants.FIELDS].sum()

            labels = [title, "Nonland",
                      "Crops",
                      "Primary Vegetation", "primf", "primn",
                      "Secondary Vegetation", "secdf", "secdn",
                      "Urban",
                      "Fields", "pastr", "range"]
            parents = ["", title,
                       title,
                       title, "Primary Vegetation", "Primary Vegetation",
                       title, "Secondary Vegetation", "Secondary Vegetation",
                       title,
                       title, "Fields", "Fields"]

            values = [total + data["nonland"], data["nonland"],
                      data["crop"],
                      primary, data["primf"], data["primn"],
                      secondary, data["secdf"], data["secdn"],
                      data["urban"],
                      fields, data["pastr"], data["range"]]

            tree_params["customdata"] = self._create_hovertext(labels, parents, values, title)
            tree_params["hovertemplate"] = "%{customdata}<extra></extra>"

        assert len(labels) == len(parents)
        assert len(parents) == len(values)

        fig = go.Figure(
            go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                **tree_params
            )
        )
        colors = px.colors.qualitative.Plotly
        fig.update_layout(
            treemapcolorway=[colors[1], colors[4], colors[2], colors[7], colors[3], colors[0]],
            margin={"t": 0, "b": 0, "l": 10, "r": 10}
        )
        return fig

    def create_pie(self, data=pd.Series, type_context=True, year=2021) -> go.Figure:
        """
        Creates a pie chart from the given data
        :param data: Pandas series of land use data
        :param type_context: If the title should be context or prescribed
        :return: Pie chart figure
        """

        values = None

        # Sum for case where all zeroes, which allows us to display pie even when presc is reset
        if data.empty or data.sum() == 0:
            values = [0 for _ in range(len(app_constants.CHART_COLS))]
            values[-1] = 1

        else:
            values = data[app_constants.CHART_COLS].tolist()

        assert len(values) == len(app_constants.CHART_COLS)

        title = f"Context in {year}" if type_context else f"Prescribed for {year+1}"

        # Attempt to match the colors from the treemap
        plo = px.colors.qualitative.Plotly
        dar = px.colors.qualitative.Dark24
        # ['crop', 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban', 'nonland]
        colors = [plo[4], plo[0], plo[2], dar[14], plo[5], plo[7], dar[2], plo[3], plo[1]]
        fig = go.Figure(
            go.Pie(
                values=values,
                labels=app_constants.CHART_COLS,
                textposition="inside",
                sort=False,
                marker_colors=colors,
                hovertemplate="%{label}<br>%{value}<br>%{percent}<extra></extra>",
                title=title
            )
        )

        # Remove the legend from the left plot so that we don't have 2
        if type_context:
            fig.update_layout(showlegend=False)

        fig.update_layout(margin={"t": 0, "b": 0, "l": 0, "r": 0})

        return fig
