import pandas as pd
from sklearn.metrics import mean_absolute_error

from predictors.predictor import Predictor

def evaluate_predictor(predictor: Predictor, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluates a given prescriptor object on a test set.
    :param predictor: a predictor object.
    :param X_test: the input data to predict on.
    :param y_test: the target data to predict.
    :return: the mean absolute error of the predictor on the test set.
    """
    return mean_absolute_error(y_test, predictor.predict(X_test))
