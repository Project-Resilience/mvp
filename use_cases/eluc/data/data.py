import xarray as xr
import regionmask
import warnings

from unileaf_util.framework.transformers.data_encoder import DataEncoder

import constants

class ELUCData():

    def import_data(self, path, update_path):
        raw = None
        # TODO: This is a bit of a hack because I'm not sure how to handle the dask warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = xr.open_zarr(path, consolidated=True, chunks="auto")

            # Get updated ELUC
            if update_path:
                eluc = xr.open_dataset(update_path)
                raw = raw.drop_vars(["ELUC", "cell_area"])
                raw = raw.merge(eluc)

            # Shift actions back a year
            raw_diffs = ['c3ann', 'c3nfx', 'c3per','c4ann', 'c4per', 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban']
            raw_diffs = [f"{col}_diff" for col in raw_diffs]
            raw[raw_diffs] = raw[raw_diffs].shift(time=-1)

            # Old time shifting
            # raw['ELUC'] = raw['ELUC'].shift(time=1)
            # raw['ELUC_diff'] = raw['ELUC_diff'].shift(time=1)
            # raw['time'] = raw.time - 1
            # assert(list(np.unique(raw.time)) == list(range(1849, 2022)))
            # mask = raw["ELUC_diff"].isnull().compute()
            # raw = raw.where(~mask, drop=True)

            country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(raw)
            raw["country"] = country_mask
        return raw
    
    def __init__(self, path, update_path, start_year=1851, test_year=2012, end_year=2022, countries=None, merge_crop=False):
        assert start_year < test_year and test_year < end_year
        raw = self.import_data(path, update_path)
        df = self.da_to_df(raw, start_year, end_year, countries, merge_crop)
        self.train_df = df.loc[:test_year]
        self.test_df = df.loc[test_year:]
        
        self.encoder = DataEncoder(self.get_fields(), constants.CAO_MAPPING)
        self.encoded_train_df = None
        self.encoded_test_df = None


    def subset_countries(self, df, countries):
        """
        Subsets dataframe by country list
        """
        idx = constants.COUNTRIES_DF[constants.COUNTRIES_DF["abbrevs"].isin(countries)].index.values
        return df[df["country"].isin(idx)].copy()


    def da_to_df(self, da, start_year=None, end_year=None, countries=None, merge_crop=False):
        df = da.to_dataframe()
        df = df.dropna()

        df = df.reorder_levels(["time", "lat", "lon"]).sort_index()

        if start_year:
            df = df.loc[start_year:]
        if end_year:
            df = df.loc[:end_year]
        if countries:
            df = self.subset_countries(df, countries)

        df["time"] = df.index.get_level_values("time")
        df["lat"] = df.index.get_level_values("lat")
        df["lon"] = df.index.get_level_values("lon")

        # Merge crops into one column because of BLUE model
        if merge_crop:
            df["crop"] = df[constants.CROP_COLS].sum(axis=1)
            df["crop_diff"] = df[[f"{c}_diff" for c in constants.CROP_COLS]].sum(axis=1)
            
        df['country_name'] = constants.COUNTRIES_DF.loc[df['country'], 'names'].values
            
        return df
    

    def get_fields(self):
        fields_df = self.train_df[constants.CAO_MAPPING["context"] + constants.CAO_MAPPING["actions"] + ["ELUC"]].astype("float64")
        fields = dict()
        # TODO: Right now this doesn't work because we don't have separate CAO mappings for merged and not merged crops
        for col in constants.CAO_MAPPING["context"] + constants.CAO_MAPPING["actions"] + ["ELUC"]:
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

        # TODO: This is just dummy values. Will it work?
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
    
    """
    These functions are to reduce the upfront cost if we are not encoding the data.
    """
    def get_encoded_train(self):
        if self.encoded_train_df is None:
            self.encoded_train_df = self.encoder.encode_as_df(self.train_df)
        return self.encoded_train_df
    
    def get_encoded_test(self):
        if self.encoded_test_df is None:
            self.encoded_test_df = self.encoder.encode_as_df(self.test_df)
        return self.encoded_test_df