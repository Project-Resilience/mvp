import warnings

import pandas as pd
import torch

from xgboost import XGBRegressor
from data_encoder import DataEncoder

from constants import fields
from constants import cao_mapping
from constants import LAND_USE_COLS
from constants import COLS_MAP
from constants import DIFF_LAND_USE_COLS
from constants import XGBOOST_FILE_PATH
from constants import LSTM_FILE_PATH
from constants import CONTEXT_COLUMNS
from constants import ALL_DIFF_LAND_USE_COLS
from constants import XGBOOST_FEATURES

# Silence xgboost warnings
warnings.filterwarnings("ignore")

class Predictor:
    """
    Wraps XGBoost model and DataEncoder.
    To be updated later to handle LSTM/other models.
    """

    def __init__(self):
        pass

    def run_predictor(self, context: pd.DataFrame, prescribed: pd.DataFrame) -> float:
        """
        Runs predictor using context and prescription.
        """
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

        prescribed_actions_df = prescribed[LAND_USE_COLS].reset_index(drop=True) \
            - context[LAND_USE_COLS].reset_index(drop=True)
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
        def __init__(self, in_features, layers=1, hidden_dim=64, dropout=0):
            super().__init__()
            self.norm = torch.nn.LayerNorm(in_features)
            self.lstm = torch.nn.LSTM(in_features, hidden_dim, num_layers=layers, batch_first=True, dropout=dropout)
            self.ff = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = self.norm(x)
            x, _ = self.lstm(x)
            x = self.ff(x[:,-1,:])
            x = torch.squeeze(x)
            return x
    
    def __init__(self, model_path=LSTM_FILE_PATH):
        """
        :param model_path: Path to XGBoost model file
        """
        super().__init__()
        self.predictor_model = self.LSTMModel(in_features=len(CONTEXT_COLUMNS + ALL_DIFF_LAND_USE_COLS))
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
        encoded_context_df = self.encoder.encode_as_df(context_df)

        # TODO: This is yucky because we have to add primn and primf 0 diffs since our
        # prescriptor doesn't handle them.
        prescribed_actions_df = prescribed[LAND_USE_COLS].reset_index(drop=True) \
            - context_df[LAND_USE_COLS].iloc[[-1]].reset_index(drop=True)
        prescribed_actions_df.rename(COLS_MAP, inplace=True, axis=1)
        encoded_prescribed_actions_df = self.encoder.encode_as_df(prescribed_actions_df)

        # TODO: @IMPORTANT WE DONT PRESCRIBE C4PER. Does this destroy our df?
        encoded_prescribed_actions_df["primn_diff"] = 0
        encoded_prescribed_actions_df["primf_diff"] = 0
        encoded_prescribed_actions_df["c4per_diff"] = 0
        encoded_context_df["c4per"] = 0
        encoded_context_df["primn_diff"] = 0
        encoded_context_df["primf_diff"] = 0
        encoded_context_df["c4per_diff"] = 0

        encoded_context_df.loc[encoded_context_df.index[-1], DIFF_LAND_USE_COLS] = encoded_prescribed_actions_df.loc[encoded_prescribed_actions_df.index[0], DIFF_LAND_USE_COLS]
        
        df_np = encoded_context_df[CONTEXT_COLUMNS + ALL_DIFF_LAND_USE_COLS].to_numpy()
        inp = torch.from_numpy(df_np)
        inp = inp.type(torch.FloatTensor)
        inp = inp.unsqueeze(0)

        out = self.predictor_model(inp)

        return out.item()
