
from dash import html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app.constants as app_constants
from app.utils import add_nonland, EvolutionHandler
from data import constants

class FilterComponent:
    def __init__(self, handler: EvolutionHandler):
        self.handler = handler
        self.slider_ids = [f"{outcome}-slider" for outcome in constants.CAO_MAPPING["outcomes"]]
        self.slider_params = ["min", "max", "value", "marks"]

    def plot_prescriptions(self, results_df: pd.DataFrame, cand_ids: list[str]) -> go.Figure:
        reco_df = self.handler.context_actions_to_recos(results_df)
        fig = go.Figure()

        # Attempt to match the colors from the treemap
        plo = px.colors.qualitative.Plotly
        dar = px.colors.qualitative.Dark24
        # ['crop', 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban', 'nonland']
        colors = [plo[4], plo[0], plo[2], dar[14], plo[5], plo[7], dar[2], plo[3], plo[1]]

        cand_id_order = cand_ids + [cand_id for cand_id in self.handler.prescriptor_list if cand_id not in cand_ids]
        reco_df["cand_id"] = pd.Categorical(reco_df["cand_id"], categories=cand_id_order, ordered=True)
        ordered_df = reco_df.sort_values("cand_id")

        for col, color in zip(app_constants.CHART_COLS, colors):

            # Add original values to the plot.
            x = ["Original"] + list(ordered_df["cand_id"])
            y = list(ordered_df[col]) if col in reco_df.columns else list(results_df[col])
            y.insert(0, results_df[col].iloc[0])

            # Candidate ids that aren't selected are grayed out
            color_list = [color] * (len(cand_ids) + 1)
            color_list = color_list + (["gray"] * (len(cand_id_order) - len(cand_ids)))

            fig.add_trace(go.Bar(x=x, y=y, name=col, marker_color=color_list))

        fig.update_layout(barmode="stack")

        return fig

    def create_outcome_sliders(self):
        """
        Creates initial outcome sliders and lines them up with their labels.
        TODO: We need to stop hard-coding their names and adjustments.
        TODO: Add a tooltip to the sliders to show their units.
        """
        sliders = []
        for outcome in constants.CAO_MAPPING["outcomes"]:
            col_id = outcome.replace(" ", "-").replace(".", "_")
            slider = dcc.RangeSlider(
                id=f"{col_id}-slider",
                min=0,
                max=1,
                value=[0, 1],
                marks={0: f"{0:.2f}", 1: f"{1:.2f}"},
                tooltip={"placement": "bottom", "always_visible": True},
                allowCross=False,
                disabled=False
            )
            sliders.append(slider)

        # w-25 and flex-grow-1 ensures they line up
        div = html.Div(
            children=[
                html.Div(
                    className="d-flex flex-row mb-2",
                    children=[
                        html.Label(outcome, className="w-25"),
                        html.Div(slider, className="flex-grow-1")
                    ]
                )
                for outcome, slider in zip(constants.CAO_MAPPING["outcomes"], sliders)
            ]
        )
        return div

    def get_div(self):
        return html.Div(
            className="w-100 vh-50",
            children=[
                dcc.Loading(
                    type="circle",
                    children=[
                        dcc.Store(id="results-store"),
                        dcc.Graph(
                            id="prescriptions"
                        ),
                        self.create_outcome_sliders()
                    ]
                )
            ]
        )

    def register_callbacks(self, app):
        @app.callback(
            [Output(slider_id, param) for slider_id in self.slider_ids for param in self.slider_params],
            Input("results-store", "data"),
            prevent_initial_call=True
        )
        def update_filter_sliders(results_json: dict[str: list]) -> list:
            results_df = pd.DataFrame(results_json)
            total_output = []
            for outcome in constants.CAO_MAPPING["outcomes"]:
                metric_output = []
                min_val = results_df[outcome].min()
                max_val = results_df[outcome].max()
                # We need to round down for the min value and round up for the max value
                val_range = max_val - min_val
                min_val_rounded = min_val - val_range / 100
                max_val_rounded = max_val + val_range / 100
                metric_output = [
                    min_val_rounded,
                    max_val_rounded,
                    [min_val_rounded, max_val_rounded],
                    {min_val_rounded: f"{min_val_rounded:.2f}", max_val_rounded: f"{max_val_rounded:.2f}"}
                ]
                total_output.extend(metric_output)
            return total_output

        @app.callback(
            Output("prescriptions", "figure"),
            State("results-store", "data"),
            [Input(slider_id, "value") for slider_id in self.slider_ids],
            prevent_initial_call=True
        )
        def update_prescriptions(results_json: dict[str: list], *slider_ranges) -> go.Figure:
            results_df = pd.DataFrame(results_json)
            if results_df.empty:
                return go.Figure()
            total_condition = True
            for outcome, slider_range in zip(constants.CAO_MAPPING["outcomes"], slider_ranges):
                condition = (results_df[outcome] >= slider_range[0]) & (results_df[outcome] <= slider_range[1])
                total_condition &= condition
            cand_idxs = results_df[total_condition]["cand_id"].tolist()
            print(f"Filtered to {len(cand_idxs)} candidates")
            return self.plot_prescriptions(results_df, cand_idxs)
