from abc import ABC
from pathlib import Path

import pandas as pd

class Prescriptor(ABC):
    """
    Abstract class for prescriptors to allow us to experiment with different implementations.
    """

    def prescribe_land_use(self, cand_id: str, results_dir: Path, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads a candidate based off id format <generation>_<id>.
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
    