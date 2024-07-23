"""
Encoder class for the ELUC dataset.
"""
import json
from pathlib import Path

import pandas as pd

from data import constants


class ELUCEncoder():
    """
    Creates an encoder for a pandas dataset by collecting fields used for minmax scaling.
    Special case for "change" column which doesn't have to be encoded
    Special case for "diff" colums which we want to force between [-1, 1] which stretches them out.
    """
    def __init__(self, fields: dict):
        self.fields = fields

    @classmethod
    def from_pandas(cls, df: pd.DataFrame):
        """
        Records fields from a pandas dataframe.
        """
        return cls(cls.get_fields(df))

    @staticmethod
    def get_fields(df: pd.DataFrame) -> dict:
        """
        Creates fields json object for the data encoder/prescriptor.
        """
        cao_cols = constants.CAO_MAPPING["context"] + constants.CAO_MAPPING["actions"] + ["ELUC"]
        fields_df = df[cao_cols].astype("float64")
        fields = {}
        for col in cao_cols:
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
                    new_df[col] = (new_df[col] - min_val) / (max_val - min_val)
        return new_df

    def decode_as_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decodes a dataframe using the fields given in the constructor.
        Uses minmax scaling.
        """
        new_df = df.copy()
        for col in new_df.columns:
            if col in self.fields:
                min_val = self.fields[col]["range"][0]
                max_val = self.fields[col]["range"][1]
                new_df[col] = new_df[col] * (max_val - min_val) + min_val
        return new_df

    def save_fields(self, path: Path):
        """
        Saves the fields to a JSON file.
        """
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.fields, file, indent=4)

    @classmethod
    def from_json(cls, path: Path):
        """
        Loads the fields from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as file:
            fields = json.load(file)
        return cls(fields)
