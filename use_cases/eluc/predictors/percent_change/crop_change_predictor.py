"""
Our heuristic model that calculates the percent change of crops from actions and context.
"""
import pandas as pd

from prsdk.predictors.predictor import Predictor

from data import constants


class CropChangePredictor(Predictor):
    """
    Heuristic that calculates the percent change of land use from actions and context.
    """
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        No fitting required for this model.
        """

    def predict(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates amount crop changed by taking the absolute value of the crop_diff column and dividing it by the
        total land used.
        """
        # Get absolute value of crop diff
        crop_diff = context_actions_df["crop_diff"].abs()
        # Divide by sum of used land
        total_land = context_actions_df[constants.LAND_USE_COLS].sum(axis=1)
        total_land = total_land.replace(0, 1)  # Avoid division by 0
        percent_changed = crop_diff / total_land
        change_df = pd.DataFrame(percent_changed, columns=["cropchange"])
        return change_df
