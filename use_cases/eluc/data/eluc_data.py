"""
File handling the processing of data for the ELUC use case.
Dataset wraps around a pandas dataframe and provides an encoder for prescriptors to use.
"""
import os
import warnings

from datasets import load_dataset, Dataset
import pandas as pd
import regionmask
import xarray as xr

from data import constants
from data.conversion import construct_countries_df
from data.eluc_encoder import ELUCEncoder


class ELUCData():
    """
    Wrapper for pandas dataframe that separates the data into train and test sets based on the time column.
    Contains an encoder for prescriptors to use.
    Can be either loaded from raw files provided by BLUE and LUH or from the processed HuggingFace dataset.
    """
    def __init__(self, df: pd.DataFrame, start_year=1851, test_year=2012, end_year=2022, countries=None):
        if countries:
            df = self.subset_countries(df, countries)
        self.train_df = df.loc[start_year:test_year-1].copy()
        if test_year:
            self.test_df = df.loc[test_year:end_year-1].copy()
            assert self.train_df['time'].max() == self.test_df["time"].min() - 1

        self.encoder = ELUCEncoder.from_pandas(self.train_df)
        # Set encoded values to None so that we don't encode them until we need to
        self.encoded_train_df = None
        self.encoded_test_df = None

    def subset_countries(self, df, countries):
        """
        Subsets dataframe by country list.
        TODO: This currently doesn't work.
        """
        countries_df = construct_countries_df()
        idx = countries_df[countries_df["abbrevs"].isin(countries)].index.values
        return df[df["country"].isin(idx)].copy()

    @classmethod
    def from_hf(cls, start_year=1851, test_year=2012, end_year=2022, countries=None):
        """
        Loads dataframe from HuggingFace dataset to be processed by ELUCData constructor.
        """
        ds = load_dataset(constants.HF_PATH)["train"]
        df = ds.to_pandas()
        df["time_idx"] = df["time"]
        df["lat_idx"] = df["lat"]
        df["lon_idx"] = df["lon"]
        df = df.set_index(["time_idx", "lat_idx", "lon_idx"], drop=True)
        return cls(df, start_year, test_year, end_year, countries)

    @classmethod
    def from_file(cls, path, update_path, start_year, test_year, end_year, countries=None):
        """
        Loads xarray from raw files which is then converted to dataframe.
        This dataframe is then processed by the ELUCDataset constructor.
        """
        raw = cls.import_data(path, update_path)
        df = cls.da_to_df(raw)
        return cls(df, start_year, test_year, end_year, countries)

    @staticmethod
    def import_data(path, update_path):
        """
        Reads in raw data and update data and processes them into an xarray.
        Replace ELUC and cell_area columns with updated ones.
        Shift diffs back a year so they align in our CAO POV.
            Originally: land use for 2021, what changed from 2020-2021, ELUC for end of 2021
            Now: land use for 2021, what changed from 2021-2022, ELUC for end of 2021
        """
        raw = None
        # TODO: This is a bit of a hack because I'm not sure how to handle the dask warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = xr.open_zarr(path, consolidated=True, chunks="auto")

            # Get updated ELUC
            eluc = xr.open_dataset(update_path)
            raw = raw.drop_vars(["ELUC", "cell_area"])
            raw = raw.merge(eluc)

            # Shift actions back a year
            raw_diffs = ['c3ann', 'c3nfx', 'c3per', 'c4ann', 'c4per',
                         'pastr', 'primf', 'primn', 'range',
                         'secdf', 'secdn', 'urban']
            raw_diffs = [f"{col}_diff" for col in raw_diffs]
            raw[raw_diffs] = raw[raw_diffs].shift(time=-1)

            # Finds country for each cell using lat/lon coordinates
            country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(raw)
            raw["country"] = country_mask
        return raw

    @staticmethod
    def da_to_df(da: xr.DataArray) -> pd.DataFrame:
        """
        Converts an xarray DataArray to a pandas DataFrame.
        Duplicates indices into columns so we can use them as features.
        Adds country name column for easier access.
        :param da: xarray DataArray to convert.
        :return: pandas DataFrame
        """
        df = da.to_dataframe()
        df = df.dropna()

        df = df.reorder_levels(["time", "lat", "lon"]).sort_index()

        # Keep time/lat/lon in columns so we can use them as features
        df["time"] = df.index.get_level_values("time")
        df["lat"] = df.index.get_level_values("lat")
        df["lon"] = df.index.get_level_values("lon")

        # Merge crops into one column because BLUE model doesn't differentiate
        df["crop"] = df[constants.CROP_COLS].sum(axis=1)
        df["crop_diff"] = df[[f"{c}_diff" for c in constants.CROP_COLS]].sum(axis=1)

        countries_df = construct_countries_df()
        df['country_name'] = countries_df.loc[df['country'], 'names'].values

        # Drop this column we used for preprocessing (?)
        df = df.drop("mask", axis=1)

        return df

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

    def push_to_hf(self, repo_path, commit_message, token=None):
        """
        Pushes data to huggingface repo. Don't use this unless you're sure you want to update it!
        :param repo_path: Path to huggingface repo.
        """
        whole_df = pd.concat([self.train_df, self.test_df])
        # We get the indices as columns anyways so we can drop them
        whole_df = whole_df.drop(["lat", "lon", "time"], axis=1)
        ds = Dataset.from_pandas(whole_df)
        if not token:
            token = os.getenv("HF_TOKEN")
        ds.push_to_hub(repo_path, commit_message=commit_message, token=token)
