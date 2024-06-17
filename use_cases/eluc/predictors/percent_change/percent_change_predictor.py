"""
Our heuristic model that calculates the percent change of land use from actions and context.
"""
import pandas as pd

from data import constants
from predictors.predictor import Predictor

class PercentChangePredictor(Predictor):
    """
    Heuristic that calculates the percent change of land use from actions and context.
    """
    def __init__(self):
        super().__init__(constants.CAO_MAPPING["context"], constants.CAO_MAPPING["actions"], ["change"])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        No fitting required for this model.
        """

    def predict(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates percent of land changed by summing the positive land diffs and dividing them by the total
        land used.
        """
        # Sum the positive diffs
        pos_diffs = context_actions_df[context_actions_df[constants.DIFF_LAND_USE_COLS] > 0]
        percent_changed = pos_diffs[constants.DIFF_LAND_USE_COLS].sum(axis=1)
        # Divide by sum of used land
        total_land = context_actions_df[constants.LAND_USE_COLS].sum(axis=1)
        total_land = total_land.replace(0, 1) # Avoid division by 0
        percent_changed = percent_changed / total_land
        change_df = pd.DataFrame(percent_changed, columns=["change"])
        return change_df
