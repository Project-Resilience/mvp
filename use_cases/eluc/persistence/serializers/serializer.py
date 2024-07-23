"""
Abstract class responsible for defining the interface of the serializer classes.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class Serializer(ABC):
    """
    Abstract class responsible for saving and loading predictor/prescriptor models locally.
    Save and load should be compatible with each other but don't necessarily have to be the same as other models.
    Save should take an object and save it to a path.
    Load should take a path and return an object.
    """
    @abstractmethod
    def save(self, model, path: Path) -> None:
        """
        Saves a model to disk.
        """
        raise NotImplementedError("Saving not implemented")

    @abstractmethod
    def load(self, path: Path):
        """
        Takes a path and returns a model.
        """
        raise NotImplementedError("Loading not implemented")
