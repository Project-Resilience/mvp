from abc import ABC
from abc import abstractmethod
import warnings
from joblib import load

# TODO: Clean this up!
import torch

from . import constants
from predictors.ELUCNeuralNet import ELUCNeuralNet

# Silence xgboost warnings
warnings.filterwarnings("ignore")

class Predictor(ABC):
    """
    Abstract class for predictor models to inherit. Must implement the predict method.
    Constructor for predictor must not take any arguments.
    """
    def __init__(self, load_path):
        pass

    @abstractmethod
    def predict(self, input):
        """
        Input: pd.DataFrame with columns: CONTEXT_COLUMNS + DIFF_LAND_USE_COLS indexed by INDEX_COLS in constants.py
        Output: pd.DataFrame with column: ELUC
        """
        pass


class SkLearnPredictor(Predictor):
    def __init__(self, load_path):
        self.model = load(load_path)

    def predict(self, input):
        pred = self.model.predict(input[constants.DIFF_LAND_USE_COLS])
        return pred[0]
    

class NeuralNetPredictor(Predictor):
    def __init__(self, load_path):
        self.model = ELUCNeuralNet(len(constants.CONTEXT_COLUMNS + constants.DIFF_LAND_USE_COLS), [4096], True, 0)
        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()        
        self.scaler = load(load_path.split('.pt')[0] + "-scaler.joblib")

    def predict(self, input):
        scaled_input = self.scaler.transform(input[constants.CONTEXT_COLUMNS + constants.DIFF_LAND_USE_COLS])
        pred = self.model.forward(torch.tensor(scaled_input, dtype=torch.float32))
        return pred.item()

class CustomPredictor(Predictor):
    """ You fill in here: """
    