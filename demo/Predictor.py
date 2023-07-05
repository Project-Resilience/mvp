import warnings

import pandas as pd

# Silence xgboost warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
import torch
from data_encoder import DataEncoder

from constants import fields, cao_mapping, LAND_USE_COLS, COLS_MAP, DIFF_LAND_USE_COLS, XGBOOST_FILE_PATH, LSTM_FILE_PATH, CONTEXT_COLUMNS, ALL_DIFF_LAND_USE_COLS, XGBOOST_FEATURES


class Predictor:
    """
    Wraps XGBoost model and DataEncoder.
    To be updated later to handle LSTM/other models.
    """

    def __init__(self):
        pass

    def run_predictor(self, context: pd.DataFrame, prescribed: pd.DataFrame) -> float:
         pass


class XGBoostPredictor(Predictor):
    def __init__(self, model_path=XGBOOST_FILE_PATH):
        """
        :param model_path: Path to XGBoost model file
        """
        super().__init__()
        self.predictor_model = XGBRegressor()
        self.predictor_model.load_model(model_path)

        self.encoder = DataEncoder(fields, cao_mapping)


    def run_predictor(self, context: pd.DataFrame, prescribed: pd.DataFrame) -> float:
        """
        Runs predictor model.
        :param context: DataFrame of context.
        :param prescribed: DataFrame of prescribed land usage to be diffed.
        :return: Tuple of predicted ELUC and percentage of land use changed.
        """

        # TODO: Encoder deletes columns not in it. Manual add c4per
        encoded_sample_context_df = self.encoder.encode_as_df(context)

        prescribed_actions_df = prescribed[LAND_USE_COLS].reset_index(drop=True) - context[LAND_USE_COLS].reset_index(drop=True)
        prescribed_actions_df.rename(COLS_MAP, axis=1, inplace=True)

        encoded_prescribed_actions_df = self.encoder.encode_as_df(prescribed_actions_df)

        # TODO: Hacky
        encoded_prescribed_actions_df["primn_diff"] = 0
        encoded_prescribed_actions_df["primf_diff"] = 0
        encoded_prescribed_actions_df["c4per_diff"] = 0
        encoded_sample_context_df["c4per"] = 0

        context_actions = pd.concat([encoded_sample_context_df, encoded_prescribed_actions_df], axis=1)

        pred = self.predictor_model.predict(context_actions[XGBOOST_FEATURES])
        pred_df = pd.DataFrame(pred, columns=["ELUC"])
        # Decode output
        out_df = self.encoder.decode_as_df(pred_df)
        return out_df.iloc[0, 0]
    

class LSTMPredictor(Predictor):

    class LSTMModel(torch.nn.Module):
        def __init__(self, layers=2, hidden_dim=128, input_dim=25, output_dim=1):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
            self.linear = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x, _ = self.lstm(x)
            x = torch.relu(x)
            x = self.linear(x)
            x = torch.squeeze(x, dim=2)
            x = torch.mean(x, dim=-1)
            return x
    
    def __init__(self, model_path=LSTM_FILE_PATH):
        """
        :param model_path: Path to XGBoost model file
        """
        super().__init__()
        self.predictor_model = self.LSTMModel()
        self.predictor_model.load_state_dict(torch.load(model_path))
        self.predictor_model.eval()

        self.encoder = DataEncoder(fields, cao_mapping)


    def run_predictor(self, context: pd.DataFrame, prescribed: pd.DataFrame) -> float:
        """
        Runs predictor model.
        :param context: DataFrame of context.
        :param prescribed: DataFrame of prescribed land usage to be diffed.
        :return: Tuple of predicted ELUC and percentage of land use changed.
        """
        
        context_df = context.reset_index(drop=True)

        # TODO: This is yucky because we have to add primn and primf 0 diffs since our
        # prescriptor doesn't handle them.
        prescribed_actions_df = prescribed[LAND_USE_COLS].reset_index(drop=True) - context_df[LAND_USE_COLS].iloc[[-1]].reset_index(drop=True)
        prescribed_actions_df.rename(COLS_MAP, inplace=True, axis=1)

        # TODO: @IMPORTANT WE DONT PRESCRIBE C4PER. Does this destroy our df?
        prescribed_actions_df["primn_diff"] = 0
        prescribed_actions_df["primf_diff"] = 0
        prescribed_actions_df["c4per_diff"] = 0
        context_df["c4per"] = 0
        context_df["primn_diff"] = 0
        context_df["primf_diff"] = 0
        context_df["c4per_diff"] = 0

        context_df.loc[context_df.index[-1], DIFF_LAND_USE_COLS] = prescribed_actions_df.loc[prescribed_actions_df.index[0], DIFF_LAND_USE_COLS]
        
        df_np = context_df[CONTEXT_COLUMNS + ALL_DIFF_LAND_USE_COLS].to_numpy()
        input = torch.from_numpy(df_np)
        input = input.type(torch.FloatTensor)
        input = input.unsqueeze(0)

        out = self.predictor_model(input)

        return out.item()