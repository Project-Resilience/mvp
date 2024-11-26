"""
A minimal version of the ELUCData class used so we don't have to go through the hassle of installing all the libraries
that require more complex dependencies.
File handling the processing of data for the ELUC use case.
Dataset wraps around a pandas dataframe and provides an encoder for prescriptors to use.
"""
import os
import warnings

from datasets import load_dataset, Dataset
import pandas as pd

from data import constants
from data.eluc_encoder import ELUCEncoder


class MinimalELUCData():
    """
    Wrapper for pandas dataframe that separates the data into train and test sets based on the time column.
    Contains an encoder for prescriptors to use.
    Can be either loaded from raw files provided by BLUE and LUH or from the processed HuggingFace dataset.
    """
    def __init__(self, df: pd.DataFrame, start_year=1851, test_year=2012, end_year=2022):
        self.train_df = df.loc[start_year:test_year-1].copy()
        if test_year:
            self.test_df = df.loc[test_year:end_year-1].copy()
            assert self.train_df['time'].max() == self.test_df["time"].min() - 1

        self.encoder = ELUCEncoder.from_pandas(self.train_df)
        # Set encoded values to None so that we don't encode them until we need to
        self.encoded_train_df = None
        self.encoded_test_df = None

    @classmethod
    def from_hf(cls, start_year=1851, test_year=2012, end_year=2022):
        """
        Loads dataframe from HuggingFace dataset to be processed by ELUCData constructor.
        """
        ds = load_dataset(constants.HF_PATH)["train"]
        df = ds.to_pandas()
        df["time_idx"] = df["time"]
        df["lat_idx"] = df["lat"]
        df["lon_idx"] = df["lon"]
        df = df.set_index(["time_idx", "lat_idx", "lon_idx"], drop=True)
        return cls(df, start_year, test_year, end_year)

    def get_encoded_train(self):
        """
        Reduces cost of encoding data by caching the encoded version.
        """
        if self.encoded_train_df is None:
            self.encoded_train_df = self.encoder.encode_as_df(self.train_df)
        return self.encoded_train_df

    def get_encoded_test(self):
        """
        Same as above but for test data.
        """
        assert self.test_df is not None, "No test data provided."
        if self.encoded_test_df is None:
            self.encoded_test_df = self.encoder.encode_as_df(self.test_df)
        return self.encoded_test_df
