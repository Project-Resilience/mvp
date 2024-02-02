from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class Predictor(ABC):
    """
    Abstract class for predictors to inherit from.
    Predictors must be able to be fit and predict on a dataframe.
    They must also be able to be saved and loaded.
    Save and load must be compatible with each other but not necessarily with other models.
    The expected flow of the model is fit -> predict -> save -> load -> predict.
    """


    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the model to the training data.
        @param X_train: dataframe with input data
        @param y_train: series with target data
        """


    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Creates a dataframe with predictions for the input dataframe.
        @param X_test: dataframe with input data
        @return: dataframe with predictions
        """


    @abstractmethod
    def save(self, path: str):
        """
        Saves the model to a path.
        @param path: path to save the model
        """


    @abstractmethod
    def load(self, path: str):
        """
        Loads a model from a path.
        @param path: path to the model
        """