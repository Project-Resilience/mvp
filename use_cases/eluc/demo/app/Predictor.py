from abc import ABC
from abc import abstractmethod
import warnings
from joblib import load

from . import constants

# Silence xgboost warnings
warnings.filterwarnings("ignore")

class Predictor(ABC):
    """
    Abstract class for predictor models to inherit.
    """

    @abstractmethod
    def predict(self, input):
        """
        Input columns: CONTEXT_COLUMNS + DIFF_LAND_USE_COLS indexed by INDEX_COLS in constants.py
        Output columns: ELUC float
        Check output validity: scale of ELUC tC/ha caused by land use change passed in input
        """
        pass


class SkLearnPredictor(Predictor):
    def __init__(self, load_path):
        self.model = load(load_path)

    def predict(self, input):
        pred = self.model.predict(input)
        return pred[0]
    

class CustomPredictor(Predictor):
    """ You fill in here: """
    