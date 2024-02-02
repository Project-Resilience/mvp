import joblib
from sklearn.linear_model import LinearRegression

from predictor import Predictor

class LinearRegressionPredictor(Predictor):
    """
    Simple linear regression predictor.
    """
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)