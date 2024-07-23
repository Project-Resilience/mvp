"""
Persistor abstract class. Wraps a serializer and provides an interface for persisting models
(ex to HuggingFace) and loading models from a persistence location.
"""
from pathlib import Path

from abc import ABC, abstractmethod

from persistence.serializers.serializer import Serializer


class Persistor(ABC):
    """
    Abstract class for persistors to inherit from.
    """
    def __init__(self, file_serializer: Serializer):
        self.file_serializer = file_serializer

    @abstractmethod
    def persist(self, model, model_path: Path, repo_id: str, **persistence_args):
        """
        Serializes a model using the file_serializer, then uploads the model to a persistence location.
        """
        raise NotImplementedError("Persisting not implemented")

    @abstractmethod
    def from_pretrained(self, path_or_url: str, **persistence_args):
        """
        Loads a model from where it was persisted from.
        """
        raise NotImplementedError("Loading not implemented")
