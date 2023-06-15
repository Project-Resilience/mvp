import pandas as pd

from xgboost import XGBRegressor
from data_encoder import DataEncoder

from constants import fields, cao_mapping, LAND_USE_COLS, COLS_MAP, DIFF_LAND_USE_COLS, XGBOOST_FILE_PATH

class Predictor():


    def __init__(self, model_path=XGBOOST_FILE_PATH):

        self.predictor_model = XGBRegressor()
        self.predictor_model.load_model(model_path)

        self.encoder = DataEncoder(fields, cao_mapping)


    def __compute_percent_changed(self, encoded_context_actions_df):
        # Sum the absolute values, but divide by 2 to avoid double counting
        # Because positive diff is offset by negative diff
        # context_action_df[DIFF_LAND_USE_COLS].abs().sum(axis=1) / 2

        encoded_context_actions_df = encoded_context_actions_df.reset_index(drop=True)
        # Decode in order to get the signed land usage diff values
        context_action_df = self.encoder.decode_as_df(encoded_context_actions_df)

        # Sum the positive diffs
        percent_changed = context_action_df[context_action_df[DIFF_LAND_USE_COLS] > 0].sum(axis=1)
        # Land usage is only a portion of that cell, e.g 0.8. Scale back to 1
        # So that percent changed really represent the percentage of change within the land use
        # portion of the cell
        # I.e. how much of the pie chart has changed?
        percent_changed = percent_changed / context_action_df[LAND_USE_COLS].sum(axis=1)
        df = pd.DataFrame(percent_changed, columns=['Change'])
        return df


    def run_predictor(self, context: pd.DataFrame, prescribed: pd.DataFrame) -> tuple:
        encoded_sample_context_df = self.encoder.encode_as_df(context)

        prescribed_actions_df = prescribed[LAND_USE_COLS].reset_index(drop=True) - context[LAND_USE_COLS].reset_index(drop=True)
        prescribed_actions_df.rename(COLS_MAP, axis=1, inplace=True)

        encoded_prescribed_actions_df = self.encoder.encode_as_df(prescribed_actions_df)

        encoded_context_actions_df = pd.concat([encoded_sample_context_df,
                                            encoded_prescribed_actions_df],
                                        axis=1)
        
        change_df = self.__compute_percent_changed(encoded_context_actions_df)
        
        new_pred = self.predictor_model.predict(encoded_context_actions_df)
        pred_df = pd.DataFrame(new_pred, columns=["ELUC"])
        # Decode output
        out_df = self.encoder.decode_as_df(pred_df)
        return out_df.iloc[0, 0], change_df.iloc[0, 0]