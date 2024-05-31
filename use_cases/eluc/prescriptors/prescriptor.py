"""
Abstract prescriptor class to be implemented.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from huggingface_hub import snapshot_download
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
        
    @abstractmethod
    def save(self, path: Path):
        """
        Saves a prescriptor to disk.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Prescriptor":
        """
        Loads a prescriptor from disk.
        """
        raise NotImplementedError
    
    @classmethod
    def from_pretrained(cls, path_or_url: str, **hf_args) -> "Prescriptor":
        """
        Loads a model from a path or if it is not found, from a huggingface repo.
        TODO: This code is copied from predictor. We need to refactor this to avoid code duplication.
        :param path_or_url: path to the model or url to the huggingface repo.
        :param hf_args: arguments to pass to the snapshot_download function from huggingface.
        """
        path = Path(path_or_url)
        if path.exists() and path.is_dir():
            return cls.load(path)
        else:
            # TODO: Need a try except block to catch download errors
            url_path = path_or_url.replace("/", "--")
            local_dir = hf_args.get("local_dir", f"prescriptors/trained_models/{url_path}")

            if not Path(local_dir).exists() or not Path(local_dir).is_dir():
                hf_args["local_dir"] = local_dir
                snapshot_download(repo_id=path_or_url, **hf_args)

            return cls.load(Path(local_dir))
