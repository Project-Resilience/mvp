"""
Abstract prescriptor class to be implemented.
"""
from abc import ABC, abstractmethod

import pandas as pd


class Prescriptor(ABC):
    """
    Abstract class for prescriptors to allow us to experiment with different implementations.
    """
    def __init__(self, context: list[str], actions: list[str]):
        self.context = context
        self.actions = actions

    @abstractmethod
    def prescribe(self, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes in a context dataframe and prescribes actions.
        Outputs a concatenation of the context and actions.
        """
        raise NotImplementedError
