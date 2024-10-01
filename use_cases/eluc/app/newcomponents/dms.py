
from dash import html, dcc, Output, State, Input, ALL
import dash_bootstrap_components as dbc
import pandas as pd


from app.components.chart import ChartComponent
from app.utils import EvolutionHandler
from data import constants

class DMSComponent():
    def __init__(self, app_df: pd.DataFrame, handler: EvolutionHandler):
        self.chart_component = ChartComponent(app_df)

        self.app_df = app_df
        self.handler = handler

    def create_default_diff_sliders(self) -> list:
        """
        Takes a context actions df and returns a list of sliders with the values set to the DIFF_RECO_COLS.
        """
        sliders = []
        for col in constants.RECO_COLS:
            sliders.append(
                dcc.Slider(
                    id={"type": "diff-slider", "index": f"{col}"},
                    min=-1,
                    max=1,
                    value=0,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "transform": "percentSlider",  # See assets/tooltip.js
                        "template": "{value}%"
                    },
                    marks=[0],
                )
            )
        div = html.Div(
            children=[
                dbc.Row([
                    dbc.Col(html.Label(diff_col)),
                    dbc.Col(slider)
                ])
                for diff_col, slider in zip(constants.DIFF_RECO_COLS, sliders)
            ]
        )
        return div

    def get_div(self):
        return html.Div(
            className="mx-5 mb-5",
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                html.B("3. Select a policy to see the prescribed actions customize them."),
                                dcc.Dropdown(
                                    id="presc-dropdown",
                                    options=[],
                                    placeholder="Select a candidate",
                                    className="mb-3"
                                ),
                                self.create_default_diff_sliders(),
                                html.Div([
                                    html.Label("Allocated Land"),
                                    dbc.Progress(
                                        id="pbar",
                                        children=[
                                            dbc.Progress(value=100, color="success", min=0, max=600)
                                        ]
                                    )
                                ], className="w-100 mt-3 mb-3"),
                                html.Div([
                                    dbc.Button("Apply Policy", id="update-button"),
                                    dbc.Alert("Make sure exactly 100% of land is being used before updating!",
                                              id="alert", color="warning", dismissable=True, is_open=False),
                                ], className="mb-3 d-flex flex-row"),
                                dbc.Button(
                                    id="eluc",
                                    children="Predicted ELUC: 00.00",
                                    color="warning",
                                    active=True,
                                    outline=True
                                ),
                                dbc.Button(
                                    id="change",
                                    children="Land Changed: 0.00%",
                                    color="warning",
                                    active=True,
                                    outline=True
                                )
                            ]
                        ),
                        dbc.Col(
                            dcc.Graph(id="context-fig")
                        ),
                        dbc.Col(
                            dcc.Graph(id="presc-fig")
                        )
                    ]
                )
            ]
        )

    def context_presc_to_df(self, context: pd.Series, actions: pd.Series) -> pd.DataFrame:
        """
        Takes a context with all columns and a presc with RECO_COLS and returns an updated context actions df.
        This df takes the difference between the RECO_COLS in presc and context and sets the DIFF_RECO_COLS to that.
        """
        context_actions = actions.combine_first(context[constants.CAO_MAPPING["context"]])
        context_actions_df = pd.DataFrame([context_actions])
        context_actions_df[constants.NO_CHANGE_COLS] = 0  # TODO: I'm not entirely sure why this line is necessary
        return context_actions_df

    def register_callbacks(self, app):
        self.chart_component.register_update_context_chart_callback(app)
        self.chart_component.register_update_presc_chart_callback(app)

        @app.callback(
            Output({"type": "diff-slider", "index": ALL}, "value", allow_duplicate=True),
            Output({"type": "diff-slider", "index": ALL}, "min"),
            State("results-store", "data"),
            Input("presc-dropdown", "value"),
            prevent_initial_call=True
        )
        def update_presc_sliders(results_json: dict[str: list], cand_id: str) -> list:
            results_df = pd.DataFrame(results_json)
            selected = results_df[results_df["cand_id"] == cand_id]
            slider_vals = selected[constants.DIFF_RECO_COLS].iloc[0].tolist()
            min_vals = selected[constants.RECO_COLS].iloc[0].tolist()
            min_vals = [-1 * min_val for min_val in min_vals]
            return slider_vals, min_vals
        
        @app.callback(
            Output("pbar", "children"),
            [Input({"type": "diff-slider", "index": ALL}, "value")]
        )
        def update_pbar(sliders) -> int:
            total = 1 + sum(sliders)
            # TODO: This is a bit of a hack due to rounding errors.
            if total < 1.01 and total > 0.99:
                total = 1
            if total < 1:
                return dbc.Progress(value=total*100, label=f"{int(total*100)}%", max=600, color="warning", bar=True)
            elif total == 1:
                return dbc.Progress(value=100, label=f"{int(total*100)}%", max=600, color="success", bar=True)
            else:
                return [
                    dbc.Progress(value=100, label=f"100%", max=600, color="success", bar=True),
                    dbc.Progress(value=(total-1)*100, label=f"{int((total-1)*100)}%", max=600, color="danger", bar=True)
                ]
        
        @app.callback(
            Output("eluc", "children"),
            Output("eluc", "color"),
            Output("change", "children"),
            Output("change", "color"),
            Input("update-button", "n_clicks"),
            State("year-input", "value"),
            State("lat-dropdown", "value"),
            State("lon-dropdown", "value"),
            [State({"type": "diff-slider", "index": ALL}, "value")],
            prevent_initial_call=True
        )
        def update_policy(n_clicks: int, year: int, lat: float, lon: float, sliders: list[float]) -> list:
            context = self.app_df.loc[year, lat, lon]
            actions = pd.Series(sliders, index=constants.DIFF_RECO_COLS)
            context_actions_df = self.context_presc_to_df(context, actions)
            outcomes_df = self.handler.predict_metrics(context_actions_df)
            eluc = outcomes_df["ELUC"].iloc[0]
            change = outcomes_df["change"].iloc[0]

            if eluc < 0:
                eluc_color = "success"
            elif eluc > 0:
                eluc_color = "danger"
            else:
                eluc_color = "warning"

            if change < 0.2:
                change_color = "success"
            elif change > 0.5:
                change_color = "danger"
            else:
                change_color = "warning"

            return f"Predicted ELUC: {eluc:.2f}", eluc_color, f"Land Changed: {(change*100):.2f}%", change_color
