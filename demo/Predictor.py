from abc import ABC, abstractmethod
import warnings
from joblib import load

from constants import RANDOM_FOREST_PATH

# Silence xgboost warnings
warnings.filterwarnings("ignore")

class Predictor(ABC):

    @abstractmethod
    def predict(self, input):
        pass


class RandomForestPredictor(Predictor):
    def __init__(self, load_path=RANDOM_FOREST_PATH):
        self.model = load(load_path)

    def predict(self, input):
        pred = self.model.predict(input)
        return pred[0]