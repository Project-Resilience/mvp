"""
Prediction component for ELUC predictor selection and predict button.
"""
from dash import Input, State, Output, ALL
from dash import dcc
from dash import html
import pandas as pd

from app import constants as app_constants
from app import utils
from data import constants
from predictors.predictor import Predictor
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor
from predictors.sklearn.sklearn_predictor import LinearRegressionPredictor, RandomForestPredictor

class PredictionComponent:
    """
    Component in charge of handling predictor selection and predict button callback.
    """
    def __init__(self, df):
        self.df = df
        self.predictors = self.load_predictors()

    def load_predictors(self) -> dict[str, Predictor]:
        """
        Loads in predictors from disk, downloads from HuggingFace first if needed.
        TODO: Currently hard-coded to load specific predictors. We need to make this able to handle any amount!
        :return: dict of predictor name -> predictor object.
        """
        predictors = {}
        nn_path = "danyoung/eluc-global-nn"
        nn_local_dir = app_constants.PREDICTOR_PATH / nn_path.replace("/", "--")
        linreg_path = "danyoung/eluc-global-linreg"
        linreg_local_dir = app_constants.PREDICTOR_PATH / linreg_path.replace("/", "--")
        rf_path = "danyoung/eluc-global-rf"
        rf_local_dir = app_constants.PREDICTOR_PATH / rf_path.replace("/", "--")
        global_nn = NeuralNetPredictor.from_pretrained(nn_path,
                                                    local_dir=nn_local_dir)
        global_linreg = LinearRegressionPredictor.from_pretrained(linreg_path,
                                                                local_dir=linreg_local_dir)
        global_rf = RandomForestPredictor.from_pretrained(rf_path,
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
            dcc.Dropdown(list((self.predictors.keys())), list(self.predictors.keys())[0], id="pred-select", style={"width": "200px"}),
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

    def register_predictor_callbacks(self, app):
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
            context_actions_df = utils.context_presc_to_df(context, presc)

            predictor = self.predictors[predictor_name]
            eluc_df = predictor.predict(context_actions_df)
            eluc = eluc_df["ELUC"].iloc[0]
            return f"{eluc:.4f}"
