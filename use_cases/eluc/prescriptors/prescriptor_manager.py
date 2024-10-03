"""
Manages multiple Prescriptors and a Predictor. Allows the user to easily prescribe based on different Prescriptors and
then predict on a uniform Predictor in order to compare them.
Additionally handles the percent changed computation.
"""
import pandas as pd

from prsdk.predictors.predictor import Predictor
from prsdk.prescriptors.prescriptor import Prescriptor


class PrescriptorManager():
    """
    Stores many Prescriptor objects and some predictors.
    Used to uniformly prescribe and predict various Predictors.
    """
    def __init__(self, prescriptors: dict[str, Prescriptor], predictors: dict[str, Predictor]):
        """
        :param prescriptors: dict of candidate id -> Prescriptor object
        :param predictors: dict of outcome -> Predictor object
        """
        self.prescriptors = prescriptors
        self.predictors = predictors

    def prescribe(self, cand_id: str, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prescribes from a context using a specific candidate.
        """
        return self.prescriptors[cand_id].prescribe(context_df)

    def predict_metrics(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes ELUC and change for each sample in a context_actions_df.
        Returns a single dataframe with a column for each outcome
        """
        outcome_df = pd.DataFrame()
        for outcome, predictor in self.predictors.items():
            outcome_df[outcome] = predictor.predict(context_actions_df)

        return outcome_df
