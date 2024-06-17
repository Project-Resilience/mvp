"""
Prescription component in charge of the prescriptor select slider, the prescription button, and the sliders for each
recommended land-use. Also manages modal that shows the pareto front.
"""
from dash import Input, State, Output, ALL
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from app import constants as app_constants
from data import constants
from prescriptors.prescriptor_manager import PrescriptorManager
from prescriptors.nsga2.land_use_prescriptor import LandUsePrescriptor

class PrescriptionComponent():
    """
    Component in charge of handling prescriptor selection and prescribe button.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Load pareto df
        self.pareto_df = pd.read_csv(app_constants.PARETO_CSV_PATH)
        self.pareto_df.sort_values(by="change", inplace=True)
        self.prescriptor_list = list(self.pareto_df["id"])

        # Load prescriptors
        self.prescriptor_manager = self.load_prescriptors()

    def load_prescriptors(self) -> tuple[list[str], PrescriptorManager]:
        """
        Loads in prescriptors from disk, downloads from HuggingFace first if needed.
        TODO: Currently hard-coded to load specific prescriptors from pareto path.
        :return: dict of prescriptor name -> prescriptor object.
        """
        prescriptors = {}
        pareto_df = pd.read_csv(app_constants.PARETO_CSV_PATH)
        pareto_df = pareto_df.sort_values(by="change")
        for cand_id in pareto_df["id"]:
            cand_path = f"danyoung/eluc-{cand_id}"
            cand_local_dir = app_constants.PRESCRIPTOR_PATH / cand_path.replace("/", "--")
            prescriptors[cand_id] = LandUsePrescriptor.from_pretrained(cand_path, local_dir=cand_local_dir)

        prescriptor_manager = PrescriptorManager(prescriptors, None)

        return prescriptor_manager

    def get_presc_select_div(self):
        """
        Slider that allows the user to select specific prescriptors along the pareto.
        """
        margin_style = {"margin-top": "-10px"}
        presc_select_div = html.Div([
            html.P("Minimize change", style={"grid-column": "1"}),
            html.Div([
                dcc.Slider(id='presc-select',
                        min=0, max=len(self.prescriptor_list)-1, step=1,
                        value=app_constants.DEFAULT_PRESCRIPTOR_IDX,
                        included=False,
                        marks={i : "" for i in range(len(self.prescriptor_list))})
            ], style={"grid-column": "2", "width": "100%", "margin-top": "8px"}),
            html.P("Minimize ELUC", style={"grid-column": "3", "padding-right": "10px"}),
            html.Button("Prescribe", id='presc-button', n_clicks=0, style={"grid-column": "4", **margin_style}),
            html.Button("View Pareto", id='pareto-button', n_clicks=0, style={"grid-column": "5", **margin_style}),
            dbc.Modal(
                    [
                        dbc.ModalHeader("Pareto front"),
                        dcc.Graph(
                            id='pareto-fig',
                            figure=self.create_pareto(
                                pareto_df=self.pareto_df,
                                presc_id=self.prescriptor_list[app_constants.DEFAULT_PRESCRIPTOR_IDX]
                            )
                        ),
                    ],
                    id="pareto-modal",
                    is_open=False,
                ),
        ], style={"display": "grid",
                  "grid-template-columns": "auto 1fr auto auto",
                  "width": "100%",
                  "align-content": "center"})

        return presc_select_div

    def register_select_prescriptor_callback(self, app):
        """
        Registers callback for clicking the Prescribe button after adjusting the prescriptor select slider.
        This updates the prescribed land-use slider values.
        """
        @app.callback(
            Output({"type": "presc-slider", "index": ALL}, "value", allow_duplicate=True),
            Input("presc-button", "n_clicks"),
            State("presc-select", "value"),
            State("year-input", "value"),
            State("lat-dropdown", "value"),
            State("lon-dropdown", "value"),
            prevent_initial_call=True
        )
        def select_prescriptor(_, presc_idx, year, lat, lon):
            """
            Selects prescriptor, runs on context, updates sliders.
            :param presc_idx: Index of prescriptor in PRESCRIPTOR_LIST to load.
            :param year: Selected context year.
            :param lat: Selected context lat.
            :param lon: Selected context lon.
            :return: Updated slider values.
            """
            presc_id = self.prescriptor_list[presc_idx]
            context = self.df.loc[year, lat, lon][constants.CAO_MAPPING["context"]]
            context_df = pd.DataFrame([context])
            prescribed = self.prescriptor_manager.prescribe(presc_id, context_df)
            # Prescribed gives it to us in diff format, we need to recompute recommendations
            for col in constants.RECO_COLS:
                prescribed[col] = context[col] + prescribed[f"{col}_diff"]
            prescribed = prescribed[constants.RECO_COLS]
            return prescribed.iloc[0].tolist()

    def register_toggle_modal_callback(self, app):
        """
        Registers the callback tha toggles the pareto modal which shows where on the pareto the current prescriptor is.
        """
        @app.callback(
            Output("pareto-modal", "is_open"),
            Output("pareto-fig", "figure"),
            [Input("pareto-button", "n_clicks")],
            [State("pareto-modal", "is_open")],
            [State("presc-select", "value")],
        )
        def toggle_modal(n, is_open, presc_idx):
            """
            Toggles pareto modal.
            :param n: Number of times button has been clicked.
            :param is_open: Whether the modal is open.
            :param presc_idx: The index of the prescriptor to show.
            :return: The new state of the modal and the figure to show.
            """
            fig = self.create_pareto(self.pareto_df, self.prescriptor_list[presc_idx])
            if n:
                return not is_open, fig
            return is_open, fig

    def create_pareto(self, pareto_df: pd.DataFrame, presc_id: int) -> go.Figure:
        """
        :param pareto_df: Pandas data frame containing the pareto front
        :param presc_id: The currently selected prescriptor id
        :return: A pareto plot figure
        """
        fig = go.Figure(
                go.Scatter(
                    x=pareto_df['change'] * 100,
                    y=pareto_df['ELUC'],
                    # marker='o',
                )
            )
        # Highlight the selected prescriptor
        presc_df = pareto_df[pareto_df["id"] == presc_id]
        fig.add_scatter(x=presc_df['change'] * 100,
                        y=presc_df['ELUC'],
                        marker={
                            "color": 'red',
                            "size": 10
                        })
        # Name axes and hide legend
        fig.update_layout(xaxis_title={"text": "Change (%)"},
                        yaxis_title={"text": 'ELUC (tC/ha)'},
                        showlegend=False,
                        title="Prescriptors",
                        )
        fig.update_traces(hovertemplate="Average Change: %{x} <span>&#37;</span>"
                                        "<br>"
                                        " Average ELUC: %{y} tC/ha<extra></extra>")
        return fig
