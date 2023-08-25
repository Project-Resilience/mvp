import os
import pandas as pd
import regionmask
import xarray as xr

from huggingface_hub import hf_hub_download

LAND_FEATURES = ['c3ann', 'c3nfx', 'c3per','c4ann', 'c4per',
 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban', 'cell_area']

LAND_DIFF_FEATURES = ['c3ann_diff', 'c3nfx_diff', 'c3per_diff','c4ann_diff', 'c4per_diff',
 'pastr_diff', 'primf_diff', 'primn_diff', 'range_diff', 'secdf_diff', 'secdn_diff', 'urban_diff']

FEATURES = LAND_FEATURES + LAND_DIFF_FEATURES
LABEL = "ELUC"

PATH_TO_DATASET = "merged_aggregated_dataset_1850_2022.zarr.zip"


def import_data(path, update_path):
    raw = xr.open_zarr(path, consolidated=True)

    # Get updated ELUC
    if update_path:
        eluc = xr.open_dataset(update_path)
        raw = raw.drop_vars(["ELUC", "cell_area"])
        raw = raw.merge(eluc)

    # Shift actions back a year
    raw[LAND_DIFF_FEATURES] = raw[LAND_DIFF_FEATURES].shift(time=-1)

    # Old time shifting
    # raw['ELUC'] = raw['ELUC'].shift(time=1)
    # raw['ELUC_diff'] = raw['ELUC_diff'].shift(time=1)
    # raw['time'] = raw.time - 1
    # assert(list(np.unique(raw.time)) == list(range(1849, 2022)))

    mask = raw["ELUC_diff"].isnull().compute()
    raw = raw.where(~mask, drop=True)

    country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(raw)
    raw["country"] = country_mask
    return raw


def da_to_df(da, countries_df):
    df = da.to_dataframe()
    df = df.dropna()
    df['country_name'] = countries_df.loc[df['country'], 'names'].values
    return df


def main():
    raw = import_data(PATH_TO_DATASET, None)
    countries_df = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.to_dataframe()
    df = da_to_df(raw, countries_df)
    df = df.loc[1982:][FEATURES + [LABEL]]
    df.to_csv("processed/eluc_1982.csv", index=True)


if __name__ == "__main__":
    main()