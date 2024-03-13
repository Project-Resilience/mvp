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

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the model to the training data.
        :param X_train: DataFrame with input data:
            The input data consists of a DataFrame with columns found in the ELUC huggingface repo.
            It is up to the model to decide which columns to use.
        :param y_train: series with target data
        """


    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a DataFrame with predictions for the input DataFrame.
        The Predictor model is expected to keep track of the label so that it can label the output
        DataFrame properly.
        :param X_test: DataFrame with input data
        :return: DataFrame with predictions
        """


    @abstractmethod
    def save(self, path: str):
        """
        Saves the model to a path.
        :param path: path to save the model
        """


    @abstractmethod
    def load(self, path: str):
        """
        Loads a model from a path.
        :param path: path to the model
        """