from sklearn.ensemble import RandomForestRegressor

from predictors.SKLearn.SKLearnPredictor import SKLearnPredictor

class RandomForestPredictor(SKLearnPredictor):
    """
    Simple linear regression predictor.
    See SKLearnPredictor for more details.
    """
    def __init__(self, features=None, **kwargs):
        super().__init__(features)
        self.model = RandomForestRegressor(**kwargs)
