from pathlib import Path

from abc import ABC, abstractmethod

class Persistor(ABC):
    """
    Abstract class for persistors to inherit from.
    """
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