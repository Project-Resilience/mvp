from sklearn.linear_model import LinearRegression

from predictors.SKLearn.SKLearnPredictor import SKLearnPredictor

class LinearRegressionPredictor(SKLearnPredictor):
    """
    Simple linear regression predictor.
    See SKLearnPredictor for more details.
    """
    def __init__(self, features=None, **kwargs):
        super().__init__(features)
        self.model = LinearRegression(**kwargs)
