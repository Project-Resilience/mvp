"""
Evaluates our predictors.
"""
from sklearn.metrics import mean_absolute_error

from data.eluc_data import ELUCData
from predictors.predictor import Predictor

class PredictorEvaluator():
    """
    Evaluator class that evaluates any Predictor object on a given test set.
    """
    def __init__(self, test_start_year=2012, test_end_year=2022, test_countries=None):
        """
        Initializes the evalutor with a test set to consistently test on.
        """
        dataset = ELUCData.from_hf(start_year=test_start_year-1,
                                   test_year=test_start_year,
                                   end_year=test_end_year,
                                   countries=test_countries)
        self.X_test = dataset.test_df.drop("ELUC", axis=1)
        self.y_test = dataset.test_df["ELUC"]

    def evaluate_predictor(self, predictor: Predictor) -> float:
        """
        Evaluates a given prescriptor object on a test set.
        :param predictor: a predictor object.
        :param X_test: the input data to predict on.
        :param y_test: the target data to predict.
        :return: the mean absolute error of the predictor on the test set.
        """
        return mean_absolute_error(self.y_test, predictor.predict(self.X_test))
