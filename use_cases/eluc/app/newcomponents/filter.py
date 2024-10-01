
from dash import html, dcc, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app.constants as app_constants
from app.utils import add_nonland, EvolutionHandler
from data import constants

class FilterComponent:
    def __init__(self, handler: EvolutionHandler):
        self.handler = handler
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

        fig.update_layout(
            barmode="stack",
            title={
                "text": "Filtered Prescriptions",
                "x": 0.5,
                "xanchor": "center"
            },
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
        )

        return fig

    def create_outcome_sliders(self):
        """
        Creates initial outcome sliders and lines them up with their labels.
        TODO: We need to stop hard-coding their names and adjustments.
        TODO: Add a tooltip to the sliders to show their units.
        """
        sliders = []
        for outcome in constants.CAO_MAPPING["outcomes"]:
            slider = dcc.RangeSlider(
                id={"type": "filter-slider", "index": outcome},
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
                    className="d-flex flex-row w-100",
                    children=[
                        html.Label(outcome, className="w-25"),
                        html.Div(slider, className="flex-grow-1")
                    ]
                ) for outcome, slider in zip(constants.CAO_MAPPING["outcomes"], sliders)
            ]
        )
        return div

    def get_div(self):
        return html.Div(
            className="mb-5 mx-5",
            children=[
                dcc.Loading(
                    type="circle",
                    children=[
                        dbc.Row(
                            className="mb-3",
                            children=[
                                dbc.Col(
                                    children=[
                                        html.B("2. Filter by desired outcomes."),
                                        html.Div(
                                            className="d-flex flex-row",
                                            children=[
                                                dbc.Button(
                                                    "Low ELUC",
                                                    id={"type": "preset-button", "index": "low-eluc"},
                                                    color="success",
                                                    className="me-1"
                                                ),
                                                dbc.Button(
                                                    "Medium",
                                                    id={"type": "preset-button", "index": "medium"},
                                                    color="warning",
                                                    className="me-1"
                                                ),
                                                dbc.Button(
                                                    "Low Change",
                                                    id={"type": "preset-button", "index": "low-change"},
                                                    color="danger",
                                                    className="me-1"
                                                )
                                            ]
                                        )
                                    ],
                                    width={"offset": 3, "size": 3}
                                ),
                                dbc.Col([
                                    self.create_outcome_sliders(),
                                ], width=4)
                            ]
                        ),
                        dcc.Store(id="results-store"),
                        dcc.Graph(
                            id="prescriptions"
                        )
                    ]
                )
            ]
        )

    def register_callbacks(self, app):
        @app.callback(
            [Output({"type": "filter-slider", "index": ALL}, param, allow_duplicate=True) for param in self.slider_params],
            Input("results-store", "data"),
        )
        def update_filter_sliders(results_json: dict[str: list]) -> list:
            results_df = pd.DataFrame(results_json)
            outputs = [[] for _ in range(len(self.slider_params))]
            for outcome in constants.CAO_MAPPING["outcomes"]:
                min_val = results_df[outcome].min()
                max_val = results_df[outcome].max()
                # We need to round down for the min value and round up for the max value
                val_range = max_val - min_val
                min_val_rounded = min_val - val_range / 100
                max_val_rounded = max_val + val_range / 100
                outputs[0].append(min_val_rounded)
                outputs[1].append(max_val_rounded)
                outputs[2].append([min_val_rounded, max_val_rounded])
                outputs[3].append({min_val_rounded: f"{min_val_rounded:.2f}",
                                   max_val_rounded: f"{max_val_rounded:.2f}"})
            return outputs

        @app.callback(
            Output({"type": "filter-slider", "index": ALL}, "value", allow_duplicate=True),
            Input({"type": "preset-button", "index": ALL}, "n_clicks"),
            State("results-store", "data"),
            prevent_inital_call=True
        )
        def select_preset(n_clicks, results_json):
            results_df = pd.DataFrame(results_json)

            elucs = results_df["ELUC"]
            low_eluc = elucs.min() + (elucs.max() - elucs.min()) / 3
            high_eluc = elucs.max() - (elucs.max() - elucs.min()) / 3

            changes = results_df["change"]
            low_change = changes.min() + (changes.max() - changes.min()) / 3
            high_change = changes.max() - (changes.max() - changes.min()) / 3

            trigger_idx = ctx.triggered_id["index"]
            if trigger_idx == "low-eluc":
                return [[elucs.min(), low_eluc], [changes.min(), changes.max()]]

            elif trigger_idx == "medium":
                return [[low_eluc, high_eluc], [low_change, high_change]]

            else:
                return [[elucs.min(), elucs.max()], [changes.min(), low_change]]

        @app.callback(
            Output("prescriptions", "figure"),
            Output("presc-dropdown", "options"),
            State("results-store", "data"),
            Input({"type": "filter-slider", "index": ALL}, "value")
        )
        def update_prescriptions(results_json: dict[str: list], slider_ranges) -> go.Figure:
            results_df = pd.DataFrame(results_json)
            if results_df.empty:
                return go.Figure()
            total_condition = True
            for outcome, slider_range in zip(constants.CAO_MAPPING["outcomes"], slider_ranges):
                condition = (results_df[outcome] >= slider_range[0]) & (results_df[outcome] <= slider_range[1])
                total_condition &= condition
            cand_idxs = results_df[total_condition]["cand_id"].tolist()
            print(f"Filtered to {len(cand_idxs)} candidates")
            return self.plot_prescriptions(results_df, cand_idxs), cand_idxs
