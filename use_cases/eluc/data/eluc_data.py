"""
Objects used to handle preprocessing the ELUC dataset.

ELUCEncoder performs simple min/max scaling on numerical columns.

AbstractData contains the member variables and methods usable by the user.
RawELUCData is an implementation of AbstractData that loads the ELUC data via.
the raw files and processes it.
ELUCData is the standard implementation of AbstractData that loads the ELUC
data from the HuggingFace repo.
"""
import warnings
import os
from abc import ABC

import xarray as xr
import regionmask
import pandas as pd
from datasets import load_dataset, Dataset

from data import constants
from data.conversion import construct_countries_df

class ELUCEncoder():
    """
    Encodes our ELUC dataset by using minmax scaling.
    """
    def __init__(self, fields):
        self.fields = fields

    def encode_as_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes a dataframe using the fields given in the constructor.
        Uses minmax scaling.
        """
        new_df = df.copy()
        for col in new_df.columns:
            if col in self.fields:
                min_val = self.fields[col]["range"][0]
                max_val = self.fields[col]["range"][1]
                # If min and max are the same, just set value to 0
                if min_val == max_val:
                    new_df[col] = 0
                else:
                    new_df[col] = (new_df[col] - self.fields[col]["range"][0]) / (self.fields[col]["range"][1] - self.fields[col]["range"][0])
        return new_df
    
    def decode_as_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decodes a dataframe using the fields given in the constructor.
        Uses minmax scaling.
        """
        new_df = df.copy()
        for col in new_df.columns:
            if col in self.fields:
                new_df[col] = new_df[col] * (self.fields[col]["range"][1] - self.fields[col]["range"][0]) + self.fields[col]["range"][0]
        return new_df


class AbstractData(ABC):
    """
    Abstract class for handling data.
    ELUCData and RawELUCData instantiate this based on different sources and create dataframes.
        ELUCData: HuggingFace repo
        RawELUCData: Raw zarr files
    Allows user to subset by countries, encode data, and potentially push to HuggingFace.
    """
    def __init__(self):
        self.countries_df = construct_countries_df()
        self.train_df = None
        self.test_df = None
        self.encoded_train_df = None
        self.encoded_test_df = None
        self.encoder = None

    def subset_countries(self, df, countries):
        """
        Subsets dataframe by country list
        """
        idx = self.countries_df[self.countries_df["abbrevs"].isin(countries)].index.values
        return df[df["country"].isin(idx)].copy()

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
        if self.encoded_test_df is None:
            self.encoded_test_df = self.encoder.encode_as_df(self.test_df)
        return self.encoded_test_df
    
    def get_fields(self) -> dict:
        """
        Creates fields json object for the data encoder/prescriptor.
        """
        fields_df = self.train_df[constants.CAO_MAPPING["context"] + constants.CAO_MAPPING["actions"] + ["ELUC"]].astype("float64")
        fields = dict()
        for col in constants.CAO_MAPPING["context"] + constants.CAO_MAPPING["actions"] + ["ELUC"]:
            # Set range of land and diff land uses manually to their true ranges because they
            # do not need to be scaled
            if col in constants.LAND_USE_COLS:
                ran = [0, 1]
            elif col in constants.DIFF_LAND_USE_COLS:
                ran = [-1, 1]
            else:
                ran = [fields_df[col].min(), fields_df[col].max()]
            fields[col] = {
                "data_type": "FLOAT",
                "has_nan": False,
                "mean": fields_df[col].mean(),
                "range": ran,
                "std_dev": fields_df[col].std(),
                "sum": fields_df[col].sum(),
                "valued": "CONTINUOUS"
            }

        # These are just dummy values so that the prescriptor knows we have a change outcome
        fields["change"] = {
            "data_type": "FLOAT",
            "has_nan": False,
            "mean": 0.5,
            "range": [0, 1],
            "std_dev": 0.1,
            "sum": len(fields_df) // 2,
            "valued": "CONTINUOUS"
        }

        return fields 
    
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
        

class ELUCData(AbstractData):
    """
    Loads ELUC data from HuggingFace repo and processes it.
    """
    
    def __init__(self, start_year=1851, test_year=2012, end_year=2022, countries=None):
        """
        If update_path is given, load raw data the old way using 2 files that are merged.
        Otherwise, path is taken to be a huggingface repo and we load the data from there.
        """
        assert start_year < test_year and test_year < end_year

        super().__init__()

        df = self.hf_to_df(constants.HF_PATH)
        if countries:
            df = self.subset_countries(df, countries)

        self.train_df = df.loc[start_year:test_year-1]
        self.test_df = df.loc[test_year:end_year-1]
        
        self.encoder = ELUCEncoder(self.get_fields())

    def hf_to_df(self, hf_repo):
        """
        Loads dataset from huggingface, converts to pandas, then sets indices appropriately to time/lat/lon.
        Keep old time/lat/lon columns so we can use them as features later.
        """
        ds = load_dataset(hf_repo)["train"]
        df = ds.to_pandas()
        df = df.set_index(["time", "lat", "lon"], drop=False)
        return df


class RawELUCData(AbstractData):
    """
    Takes in the raw ELUC data files and processes it.
    """
    def __init__(self, path, update_path, start_year=1851, test_year=2012, end_year=2022, countries=None):
        super().__init__()
        raw = self.import_data(path, update_path)
        df = self.da_to_df(raw, start_year, end_year, countries)

        self.train_df = df.loc[start_year:test_year-1]
        self.test_df = df.loc[test_year:end_year-1]
        
        self.encoder = ELUCEncoder(self.get_fields())

    def import_data(self, path, update_path):
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
            raw_diffs = ['c3ann', 'c3nfx', 'c3per','c4ann', 'c4per', 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban']
            raw_diffs = [f"{col}_diff" for col in raw_diffs]
            raw[raw_diffs] = raw[raw_diffs].shift(time=-1)

            # Finds country for each cell using lat/lon coordinates
            country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(raw)
            raw["country"] = country_mask
        return raw
    
    def da_to_df(self, da: xr.DataArray, start_year=None, end_year=None, countries=None) -> pd.DataFrame:
        """
        Converts an xarray DataArray to a pandas DataFrame.
        Duplicates indices into columns so we can use them as features.
        Adds country name column for easier access.
        :param da: xarray DataArray to convert.
        :param start_year: Year to start at (inclusive)
        :param end_year: Year to end at (uninclusive)
        :param countries: List of country abbreviations to subset by
        :param merge_crop: Whether to merge crop columns into one column.
            (Note: Still leaves crop types untouched, just adds merged crop column)
        :return: pandas DataFrame
        """
        df = da.to_dataframe()
        df = df.dropna()

        df = df.reorder_levels(["time", "lat", "lon"]).sort_index()

        if start_year:
            df = df.loc[start_year:]
        if end_year:
            df = df.loc[:end_year]
        if countries:
            df = self.subset_countries(df, countries)

        # Keep time/lat/lon in columns so we can use them as features
        df["time"] = df.index.get_level_values("time")
        df["lat"] = df.index.get_level_values("lat")
        df["lon"] = df.index.get_level_values("lon")

        # Merge crops into one column because BLUE model doesn't differentiate
        df["crop"] = df[constants.CROP_COLS].sum(axis=1)
        df["crop_diff"] = df[[f"{c}_diff" for c in constants.CROP_COLS]].sum(axis=1)
            
        df['country_name'] = self.countries_df.loc[df['country'], 'names'].values
        
        # Drop this column we used for preprocessing (?)
        df = df.drop("mask", axis=1)
            
        return df
