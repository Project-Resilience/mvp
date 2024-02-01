from abc import ABC, abstractmethod

import pandas as pd

class Predictor(ABC):

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the model to the training data.
        @param X_train: dataframe with input data
        @param y_train: series with target data
        """


    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a dataframe with predictions for the input dataframe.
        @param X_test: dataframe with input data
        @return: dataframe with predictions
        """
        pass


    @abstractmethod
    def save(self, path: str):
        """
        Saves the model to a path.
        @param path: path to save the model
        """
        pass


    @abstractmethod
    def load(self, path: str):
        """
        Loads a model from a path.
        @param path: path to the model
        """
        pass