from abc import ABC
from pathlib import Path

import pandas as pd

from data import constants

class Prescriptor(ABC):
    """
    Abstract class for prescriptors to allow us to experiment with different implementations.
    """

    def prescribe_land_use(self, context_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Loads a candidate prescriptor using kwargs.
        Then takes in a context dataframe, and prescribes actions.
        Outputs a concatenation of the context and actions.
        """
        raise NotImplementedError
    
    def predict_metrics(self, context_actions_df: pd.DataFrame) -> tuple:
        """
        Takes in a context actions dataframe and uses the predictor the prescriptor
        was trained on to predict ELUC. Then computes change.
        Returns a dataframe of ELUC and change.
        """
        raise NotImplementedError
    
    def _compute_percent_changed(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates percent of land changed by prescriptor.
        """
        # Sum the positive diffs
        percent_changed = context_actions_df[context_actions_df[constants.DIFF_LAND_USE_COLS] > 0][constants.DIFF_LAND_USE_COLS].sum(axis=1)
        # Divide by sum of used land
        total_land = context_actions_df[constants.LAND_USE_COLS].sum(axis=1)
        total_land = total_land.replace(0, 1) # Avoid division by 0
        percent_changed = percent_changed / total_land
        change_df = pd.DataFrame(percent_changed, columns=["change"])
        return change_df
    