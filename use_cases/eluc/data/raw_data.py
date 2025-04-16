"""
Handles the raw preprocessing of the data. This is extracted from the ELUCData class so that the user doesn't have
to worry about installation of xarray, regionmask, GDAL, etc.
This is not needed for most users.
"""
import warnings

import pandas as pd
import regionmask
import xarray as xr

from data import constants
from data.conversion import construct_countries_df
from data.eluc_data import ELUCData


class RawData(ELUCData):
    """
    Can do everything ELUCData does, but also has functions to load the raw data from file.
    """
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
