"""
See here for how to impelement a predictor:
"""
import pandas as pd

from data import constants
from predictors.predictor import Predictor

class TemplatePredictor(Predictor):
    """
    A template predictor returning dummy values for ELUC and change.
    The class that gets passed into the Evaluator should call the load method which should return a Predictor.
    The Predictor just needs to impelement predict.
    """
    def __init__(self):
        super().__init__(context=constants.CAO_MAPPING["context"],
                         actions=constants.CAO_MAPPING["actions"],
                         outcomes=constants.CAO_MAPPING["outcomes"])

    def fit(self, X_train, y_train):
        pass

    def predict(self, context_actions_df):
        dummy_eluc = list(range(len(context_actions_df)))
        dummy_change = list(range(len(context_actions_df), 0, -1))
        return pd.DataFrame({"ELUC": dummy_eluc, "change": dummy_change}, index=context_actions_df.index)
    
    @classmethod
    def load(cls, path) -> "TemplatePredictor":
        print("Loading model from", path)
        return cls()