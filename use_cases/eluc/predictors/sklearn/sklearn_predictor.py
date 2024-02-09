import json

from pathlib import Path
from abc import ABC

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from predictors.predictor import Predictor

class SKLearnPredictor(Predictor, ABC):
    """
    Simple abstract class for sklearn predictors.
    Keeps track of features fit on.
    """
    def __init__(self, features=None, label=None, **kwargs):
        self.features = features
        self.label = label
        self.model = None

    def save(self, path: str):
        """
        Saves saves model and features into format for loading.
        Generates path to folder if it does not exist.
        :param path: path to folder to save model files.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        config = {
            "features": self.features,
            "label": self.label
        }
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        joblib.dump(self.model, save_path / "model.joblib")

    def load(self, path):
        """
        Loads saved model and features from a folder.
        :param path: path to folder to load model files from.
        """
        load_path = Path(path)
        with open(load_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            self.features = config["features"]
            self.label = config["label"]
        self.model = joblib.load(load_path / "model.joblib")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits SKLearn model with standard sklearn fit method.
        :param X_train: DataFrame with input data
        :param y_train: series with target data
        """
        if self.features:
            X_train = X_train[self.features]
        else:
            self.features = list(X_train.columns)
        self.label = y_train.name
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Standard sklearn predict method.
        Makes sure to use the same features as were used in fit.
        :param X_test: DataFrame with input data
        :return: properly labeled DataFrame with predictions and matching index.
        """
        if self.features:
            X_test = X_test[self.features]
        y_pred = self.model.predict(X_test)
        return pd.DataFrame(y_pred, index=X_test.index, columns=[self.label])
    
class LinearRegressionPredictor(SKLearnPredictor):
    """
    Simple linear regression predictor.
    See SKLearnPredictor for more details.
    """
    def __init__(self, features=None, **kwargs):
        super().__init__(features)
        self.model = LinearRegression(**kwargs)
    
class RandomForestPredictor(SKLearnPredictor):
    """
    Simple linear regression predictor.
    See SKLearnPredictor for more details.
    """
    def __init__(self, features=None, **kwargs):
        super().__init__(features)
        self.model = RandomForestRegressor(**kwargs)
        