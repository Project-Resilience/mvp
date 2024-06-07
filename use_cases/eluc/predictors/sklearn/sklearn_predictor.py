"""
Abstract SKLearn predictor and its implementations.
Since the SKLearn library is standardized we can easily make more.
"""
import json

from pathlib import Path
from abc import ABC

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from data import constants
from predictors.predictor import Predictor

class SKLearnPredictor(Predictor, ABC):
    """
    Simple abstract class for sklearn predictors.
    Keeps track of features fit on and label to predict.
    """
    def __init__(self, model_config: dict):
        """
        Model config contains the following:
        features: list of features to use for prediction (optional, defaults to all features)
        label: name of the label to predict (optional, defaults to passed label during fit)
        Any other parameters are passed to the model.
        """
        super().__init__(constants.CAO_MAPPING["context"], constants.CAO_MAPPING["actions"], ["ELUC"])
        self.config = model_config
        self.model = None

    def save(self, path: str):
        """
        Saves saves model and features into format for loading.
        Generates path to folder if it does not exist.
        :param path: path to folder to save model files.
        """
        if isinstance(path, str):
            save_path = Path(path)
        else:
            save_path = path
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "config.json", "w", encoding="utf-8") as file:
            json.dump(self.config, file)
        joblib.dump(self.model, save_path / "model.joblib")

    @classmethod
    def load(cls, path) -> "SKLearnPredictor":
        """
        Loads saved model and config from a local folder.
        :param path: path to folder to load model files from.
        """
        load_path = Path(path)
        if not load_path.exists() or not load_path.is_dir():
            raise FileNotFoundError(f"Path {path} does not exist.")
        if not (load_path / "config.json").exists() or not (load_path / "model.joblib").exists():
            raise FileNotFoundError("Model files not found in path.")

        with open(load_path / "config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
        sklearn_predictor = cls(config)
        sklearn_predictor.model = joblib.load(load_path / "model.joblib")
        return sklearn_predictor

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits SKLearn model with standard sklearn fit method.
        If we passed in features, use those. Otherwise use all columns.
        :param X_train: DataFrame with input data
        :param y_train: series with target data
        """
        if "features" in self.config:
            X_train = X_train[self.config["features"]]
        else:
            self.config["features"] = list(X_train.columns)
        self.config["label"] = y_train.name
        self.model.fit(X_train, y_train)

    def predict(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standard sklearn predict method.
        Makes sure to use the same features as were used in fit.
        :param context_actions_df: DataFrame with input data
        :return: properly labeled DataFrame with predictions and matching index.
        """
        context_actions_df = context_actions_df[self.config["features"]]
        y_pred = self.model.predict(context_actions_df)
        return pd.DataFrame(y_pred, index=context_actions_df.index, columns=[self.config["label"]])

class LinearRegressionPredictor(SKLearnPredictor):
    """
    Simple linear regression predictor.
    See SKLearnPredictor for more details.
    """
    def __init__(self, model_config: dict):
        if not model_config:
            model_config = {}
        super().__init__(model_config)
        lr_config = {key: value for key, value in model_config.items() if key not in ["features", "label"]}
        self.model = LinearRegression(**lr_config)

class RandomForestPredictor(SKLearnPredictor):
    """
    Simple random forest predictor.
    See SKLearnPredictor for more details.
    Overrides save method in order to compress it.
    """
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        rf_config = {key: value for key, value in model_config.items() if key not in ["features", "label"]}
        self.model = RandomForestRegressor(**rf_config)

    def save(self, path: str, compression=0):
        """
        Overrides save method to compress file since Random Forests are extremely large.
        :param path: path to folder to save model files.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "config.json", "w", encoding="utf-8") as file:
            json.dump(self.config, file)
        joblib.dump(self.model, save_path / "model.joblib", compress=compression)
        