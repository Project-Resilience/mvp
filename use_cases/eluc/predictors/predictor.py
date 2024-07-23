"""
Abstract class for predictors to inherit from.
"""
from abc import ABC, abstractmethod

import pandas as pd


class Predictor(ABC):
    """
    Abstract class for predictors to inherit from.
    Predictors must be able to be fit and predict on a DataFrame.
    It is up to the Predictor to keep track of the proper label to label the output DataFrame.
    They must also be able to be saved and loaded.
    Save and load must be compatible with each other but not necessarily with other models.
    The expected flow of the model is fit -> predict -> save -> load -> predict.
    """
    def __init__(self, context: list[str], actions: list[str], outcomes: list[str]):
        """
        Initializes the Predictor with the context, actions, and outcomes.
        :param context: list of context columns
        :param actions: list of action columns
        :param outcomes: list of outcome columns
        """
        self.context = context
        self.actions = actions
        self.outcomes = outcomes

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the model to the training data.
        :param X_train: DataFrame with input data:
            The input data consists of a DataFrame with columns found in the ELUC huggingface repo.
            It is up to the model to decide which columns to use.
        :param y_train: series with target data
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a DataFrame with predictions for the input DataFrame.
        The Predictor model is expected to keep track of the label so that it can label the output
        DataFrame properly.
        :param context_actions_df: DataFrame with input data
        :return: DataFrame with predictions
        """
        raise NotImplementedError
