"""
See here for how to impelement a predictor:
"""
import pandas as pd

from prsdk.predictors.predictor import Predictor


class TemplatePredictor(Predictor):
    """
    A template predictor returning dummy values for ELUC.
    The class that gets passed into the Evaluator should call the load method which should return a Predictor.
    The Predictor just needs to impelement predict.
    """
    def fit(self, X_train, y_train):
        pass

    def predict(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        dummy_eluc = list(range(len(context_actions_df)))
        return pd.DataFrame({"ELUC": dummy_eluc}, index=context_actions_df.index)

    @classmethod
    def load(cls, path: str) -> "TemplatePredictor":
        """
        Dummy load function that just returns a new instance of the class.
        """
        print("Loading model from", path)
        return cls()
