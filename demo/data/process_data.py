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

COUNTRY_LIST = ["GB", "BR", "CH"]


def process_ds(path):
    ds = xr.open_zarr(path, consolidated=True)

    # Shift diffs back a year instead of context/eluc up a year
    ds[LAND_DIFF_FEATURES] = ds[LAND_DIFF_FEATURES].shift(time=-1)

    country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds)
    ds = ds.assign_coords(country=country_mask)
    
    return ds


def convert_to_dataframe(da):
    df = da.to_dataframe()
    df = df.dropna()
    return df[FEATURES + [LABEL]]


def main():
    print("Processing data...")
    print("Downloading data...")
    if not os.path.exists("merged_aggregated_dataset_1850_2022.zarr.zip"):
        hf_hub_download(
            token=os.environ.get("HF_TOKEN"),
            repo_id="danyoung/eluc-dataset",
            repo_type="dataset",
            filename="merged_aggregated_dataset_1850_2022.zarr.zip",
            local_dir="./",
            local_dir_use_symlinks=False)
    print("Downloaded data")
    PATH_TO_DATASET = "merged_aggregated_dataset_1850_2022.zarr.zip"
    ds = process_ds(PATH_TO_DATASET)
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    countries_df = countries.to_dataframe()
    country_id_list = countries_df.index[countries_df['abbrevs'].isin(COUNTRY_LIST)].tolist()

    da = ds.where(ds.country.isin(country_id_list), drop=True)
    df = convert_to_dataframe(da)
    print(df.head())
    path = ""
    for c in COUNTRY_LIST:
        path += c + "_"
    path += "eluc.csv"
    print(path)
    if not os.path.isdir("processed/"):
        print("mkdir")
        os.mkdir("processed/")
    print('to csv')
    df.to_csv(f"processed/{path.lower()}", index=True)
    print('done')

if __name__ == "__main__":
    main()