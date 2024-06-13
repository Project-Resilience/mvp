"""
Abstract prescriptor class to be implemented.
"""
from abc import ABC, abstractmethod

import pandas as pd

class Prescriptor(ABC):
    """
    Abstract class for prescriptors to allow us to experiment with different implementations.
    Save and load must be compatible with each other but not necessarily with other models.
    """
    @abstractmethod
    def prescribe(self, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes in a context dataframe and prescribes actions.
        Outputs a concatenation of the context and actions.
        """
        raise NotImplementedError
