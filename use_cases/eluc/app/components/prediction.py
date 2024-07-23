"""
Prediction component for ELUC predictor selection and predict button.
"""
from math import isclose

from dash import Input, State, Output, ALL
from dash import dcc
from dash import html
import pandas as pd

from app import constants as app_constants
from data import constants
from persistence.persistors.hf_persistor import HuggingFacePersistor
from persistence.serializers.neural_network_serializer import NeuralNetSerializer
from persistence.serializers.sklearn_serializer import SKLearnSerializer
from predictors.predictor import Predictor
from predictors.percent_change.percent_change_predictor import PercentChangePredictor


class PredictionComponent:
    """
    Component in charge of handling predictor selection and predict button callback.
    """
    def __init__(self, df):
        self.df = df
        self.predictors = self.load_predictors()
        self.change_predictor = PercentChangePredictor()

    def load_predictors(self) -> dict[str, Predictor]:
        """
        Loads in predictors from disk, downloads from HuggingFace first if needed.
        TODO: Currently hard-coded to load specific predictors. We need to make this able to handle any amount!
        :return: dict of predictor name -> predictor object.
        """
        nn_persistor = HuggingFacePersistor(NeuralNetSerializer())
        sklearn_persistor = HuggingFacePersistor(SKLearnSerializer())
        predictors = {}
        nn_path = "danyoung/eluc-global-nn"
        nn_local_dir = app_constants.PREDICTOR_PATH / nn_path.replace("/", "--")
        linreg_path = "danyoung/eluc-global-linreg"
        linreg_local_dir = app_constants.PREDICTOR_PATH / linreg_path.replace("/", "--")
        rf_path = "danyoung/eluc-global-rf"
        rf_local_dir = app_constants.PREDICTOR_PATH / rf_path.replace("/", "--")
        global_nn = nn_persistor.from_pretrained(nn_path,
                                                 local_dir=nn_local_dir)
        global_linreg = sklearn_persistor.from_pretrained(linreg_path,
                                                          local_dir=linreg_local_dir)
        global_rf = sklearn_persistor.from_pretrained(rf_path,
                                                      local_dir=rf_local_dir)

        predictors["Global Neural Network"] = global_nn
        predictors["Global Linear Regression"] = global_linreg
        predictors["Global Random Forest"] = global_rf

        return predictors

    def get_predict_div(self):
        """
        HTML div for the predictor selector and predict button showing outcomes.
        """
        predict_div = html.Div([
            dcc.Dropdown(list((self.predictors.keys())),
                         list(self.predictors.keys())[0],
                         id="pred-select",
                         style={"width": "200px"}),
            html.Button("Predict", id='predict-button', n_clicks=0,),
            html.Label("Predicted ELUC:", style={'padding-left': '10px'}),
            dcc.Input(
                value="",
                type="text",
                disabled=True,
                id="predict-eluc",
            ),
            html.Label("tC/ha", style={'padding-left': '2px'}),
            html.Label("Land Change:", style={'padding-left': '10px'}),
            dcc.Input(
                value="",
                type="text",
                disabled=True,
                id="predict-change",
            ),
            html.Label("%", style={'padding-left': '2px'}),
        ], style={"display": "flex", "flex-direction": "row", "width": "90%", "align-items": "center"})
        return predict_div

    def register_predictor_callback(self, app):
        """
        Registers predict button callback to predict ELUC from context df and prescription sliders
        """
        @app.callback(
            Output("predict-eluc", "value"),
            Input("predict-button", "n_clicks"),
            State("year-input", "value"),
            State("lat-dropdown", "value"),
            State("lon-dropdown", "value"),
            State({"type": "presc-slider", "index": ALL}, "value"),
            State("pred-select", "value"),
            prevent_initial_call=True
        )
        def predict(_, year, lat, lon, sliders, predictor_name):
            """
            Predicts ELUC from context and prescription stores.
            :param year: Selected context year.
            :param lat: Selected context lat.
            :param lon: Selected context lon.
            :param sliders: Prescribed slider values.
            :param predictor_name: String name of predictor to use from dropdown.
            :return: Predicted ELUC.
            """
            context = self.df.loc[year, lat, lon]
            presc = pd.Series(sliders, index=constants.RECO_COLS)
            context_actions_df = self.context_presc_to_df(context, presc)

            predictor = self.predictors[predictor_name]
            eluc_df = predictor.predict(context_actions_df)
            eluc = eluc_df["ELUC"].iloc[0]
            return f"{eluc:.4f}"

    def register_land_use_callback(self, app):
        """
        Registers callback that automatically updates the percent land changed when sliders are changed.
        """
        @app.callback(
            Output("sum-warning", "children"),
            Output("predict-change", "value"),
            Input({"type": "presc-slider", "index": ALL}, "value"),
            State("year-input", "value"),
            State("lat-dropdown", "value"),
            State("lon-dropdown", "value"),
            State("locks", "value"),
            prevent_initial_call=True
        )
        def compute_land_change(sliders, year, lat, lon, locked):
            """
            Computes land change percent for output.
            Warns user if values don't sum to 1.
            :param sliders: Slider values to store.
            :param year: Selected context year.
            :param lat: Selected context lat.
            :param lon: Selected context lon.
            :param locked: Locked columns to check for warning.
            :return: Warning if necessary, land change percent.
            """
            context = self.df.loc[year, lat, lon]
            presc = pd.Series(sliders, index=constants.RECO_COLS)
            context_actions_df = self.context_presc_to_df(context, presc)

            warnings = []
            # Check if prescriptions sum to 1
            # TODO: Are we being precise enough?
            new_sum = presc.sum()
            old_sum = context[constants.RECO_COLS].sum()
            if not isclose(new_sum, old_sum, rel_tol=1e-7):
                warnings.append(html.P(f"WARNING: Please make sure prescriptions sum to: {str(old_sum * 100)} \
                                    instead of {str(new_sum * 100)} by clicking \"Sum to 100\""))

            # Check if sum of locked prescriptions are > sum(land use)
            # TODO: take a look at this logic.
            if locked and presc[locked].sum() > old_sum:
                warnings.append(html.P("WARNING: Sum of locked prescriptions is greater than sum of land use. \
                                    Please reduce one before proceeding"))

            # Check if any prescriptions below 0
            if (presc < 0).any():
                warnings.append(html.P("WARNING: Negative values detected. Please lower the value of a locked slider."))

            # Compute total change
            change = self.change_predictor.predict(context_actions_df)

            return warnings, f"{change['change'].iloc[0] * 100:.2f}"

    def context_presc_to_df(self, context: pd.Series, presc: pd.Series) -> pd.DataFrame:
        """
        Takes a context with all columns and a presc with RECO_COLS and returns an updated context actions df.
        This df takes the difference between the RECO_COLS in presc and context and sets the DIFF_RECO_COLS to that.
        """
        diff = presc - context[constants.RECO_COLS]
        diff = diff.rename({col: f"{col}_diff" for col in diff.index})
        context_actions = diff.combine_first(context[constants.CAO_MAPPING["context"]])
        context_actions_df = pd.DataFrame([context_actions])
        context_actions_df[constants.NO_CHANGE_COLS] = 0  # TODO: I'm not entirely sure why this line is necessary
        return context_actions_df
