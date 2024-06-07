"""
Abstract class for predictors to inherit from.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from huggingface_hub import snapshot_download
import pandas as pd


class Predictor(ABC):
    """
    Abstract class for predictors to inherit from.
    Predictors must be able to be fit and predict on a DataFrame.
    It is up to the Predictor to keep track of the proper label to label the output DataFrame.
    They must also be able to be saved and loaded.
    Save and load must be compatible with each other but not necessarily with other models.
    The expected flow of the model is fit -> predict -> save -> load -> predict.
    """
    def __init__(self, context: list[str], actions: list[str], outcomes: list[str]):
        """
        Initializes the Predictor with the context, actions, and outcomes.
        :param context: list of context columns
        :param actions: list of action columns
        :param outcomes: list of outcome columns
        """
        self.context = context
        self.actions = actions
        self.outcomes = outcomes

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the model to the training data.
        :param X_train: DataFrame with input data:
            The input data consists of a DataFrame with columns found in the ELUC huggingface repo.
            It is up to the model to decide which columns to use.
        :param y_train: series with target data
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a DataFrame with predictions for the input DataFrame.
        The Predictor model is expected to keep track of the label so that it can label the output
        DataFrame properly.
        :param context_actions_df: DataFrame with input data
        :return: DataFrame with predictions
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        """
        Saves the model to a local path.
        :param path: path to save the model
        """
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, path_or_url: str, **hf_args) -> "Predictor":
        """
        Loads a model from a path or if it is not found, from a huggingface repo.
        :param path_or_url: path to the model or url to the huggingface repo.
        :param hf_args: arguments to pass to the snapshot_download function from huggingface.
        """
        path = Path(path_or_url)
        if path.exists() and path.is_dir():
            return cls.load(path)
        # TODO: Need a try except block to catch download errors
        url_path = path_or_url.replace("/", "--")
        local_dir = hf_args.get("local_dir", f"predictors/trained_models/{url_path}")

        if not Path(local_dir).exists() or not Path(local_dir).is_dir():
            hf_args["local_dir"] = local_dir
            snapshot_download(repo_id=path_or_url, **hf_args)

        return cls.load(Path(local_dir))

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        """
        Loads a model from the path on disk.
        :param path: path to the model
        """
        raise NotImplementedError
