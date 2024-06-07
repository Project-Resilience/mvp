"""
Manages multiple Prescriptors and a Predictor. Allows the user to easily prescribe based on different Prescriptors and
then predict on a uniform Predictor in order to compare them.
Additionally handles the percent changed computation.
"""
import pandas as pd

from data import constants
from predictors.predictor import Predictor
from prescriptors.prescriptor import Prescriptor

class PrescriptorManager():
    """
    Stores many Prescriptor objects and a predictor.
    Used to uniformly prescribe and predict various Predictors.
    """
    def __init__(self, prescriptors: dict[str, Prescriptor], predictor: Predictor):
        self.prescriptors = prescriptors
        self.predictor = predictor

    def prescribe(self, cand_id: str, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prescribes from a context using a specific candidate.
        """
        return self.prescriptors[cand_id].prescribe(context_df)

    def predict_metrics(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes ELUC and change for each sample in a context_actions_df.
        """
        eluc_df = self.predictor.predict(context_actions_df)
        change_df = self.compute_percent_changed(context_actions_df)

        return eluc_df, change_df

    # TODO: Move this to its own predictor
    def compute_percent_changed(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates percent of land changed by prescriptor.
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
