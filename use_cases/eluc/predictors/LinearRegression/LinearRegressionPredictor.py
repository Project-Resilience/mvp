import joblib
import os
import json

from sklearn.linear_model import LinearRegression

from predictors.predictor import Predictor

class LinearRegressionPredictor(Predictor):
    """
    Simple linear regression predictor.
    """
    def __init__(self, features=None, **kwargs):
        self.features = features
        self.model = LinearRegression(**kwargs)

    def save(self, path: str):
        """
        Saves saves model and features into format for loading.
        Generates path to folder if it does not exist.
        :param path: path to folder to save model files.
        """
        os.makedirs(path, exist_ok=True)
        config = {
            "features": self.features,
        }
        json.dump(config, open(os.path.join(path, "config.json"), "w"))
        joblib.dump(self.model, os.path.join(path, "model.joblib"))

    def load(self, path):
        self.features = json.load(open(os.path.join(path, "config.json")))["features"]
        self.model = joblib.load(os.path.join(path, "model.joblib"))

    def fit(self, X_train, y_train):
        if self.features:
            X_train = X_train[self.features]
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.features:
            X_test = X_test[self.features]
        return self.model.predict(X_test)