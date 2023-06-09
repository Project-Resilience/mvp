
# Copyright (C) 2021-2023 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# unileaf-util SDK Software in commercial settings.
#
# END COPYRIGHT
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

UNILEAF_TO_PANDAS_TYPE = {
    "BOOL": bool,
    "INT": int,
    "FLOAT": float,
    "STRING": str
}


class DataEncoder:
    """
    Encodes and decodes LEAF datasets:
    - scales numerical data between 0 and 1
    - creates one-hot vectors for categorical data
    """

    def __init__(self, fields: Dict[str, Dict[str, Any]],
                 cao_mapping: Dict[str, List[str]]):
        """
        Creates 1 encoder per field, and fits them
        :param fields: a dictionary of field name to field attributes. Field attributes are:
        - "valued": "CONTINUOUS" or "CATEGORICAL"
        - "data_type": "INT", "STRING", "FLOAT" or "BOOL"
        - "range": an array containing the min and max values for numerical data, e.g. [0, 100]
        - "discrete_categorical_values": an array containing the list of categorical values,
         e.g. ["red", "green", "blue"]
         Example:
             "fields": {
        "Age": {
            "range": [
                0.42,
                80
            ],
            "valued": "CONTINUOUS",
            "data_type": "FLOAT",
            "discrete_categorical_values": [
                "<Column contains 88 categories. Please refine your dataset to have no more than 20 categories>"
            ]
        },
        "Sex": {
            "range": [
                0,
                0
            ],
            "valued": "CATEGORICAL",
            "data_type": "STRING",
            "discrete_categorical_values": [
                "female",
                "male"
            ]
        }
        :param cao_mapping: a dictionary with `context`, `actions` and `outcomes` keys where each key
                returns a List of the selected column names as strings. For example:
                {
                    "context": ["Age", "Sex"],
                    "actions": [],
                    "outcomes": []
                }
        """
        # Keep a dictionary of transformers by column name
        self.fields = fields
        self.cao_mapping = cao_mapping
        self.transformers = {}
        self.column_length = {}
        for field in fields:
            if self._is_categorical(field):
                # Categorical
                field_values = fields[field]["discrete_categorical_values"]
                if self._is_outcome(field):
                    # Outcome: use LabelEncoder to encode the values between 0 and n_classes-1
                    transformer = LabelEncoder()
                    self.column_length[field] = 1
                    transformer.fit(field_values)
                else:
                    # Feature: one-hot encode
                    transformer = OneHotEncoder(sparse=False)
                    self.column_length[field] = len(field_values)
                    data_df = pd.DataFrame({field: field_values})
                    if self._is_numerical(field):
                        # Convert to numeric in case the discrete categorical values where strings
                        data_df[field] = pd.to_numeric(data_df[field])
                    transformer.fit(data_df)
            else:
                # Numerical: scale it between 0 and 1
                field_values = fields[field]["range"]
                # Clip any value that comes in outside the range before encoding
                transformer = MinMaxScaler(clip=True)
                self.column_length[field] = 1
                data_df = pd.DataFrame({field: field_values})
                transformer.fit(data_df)
            self.transformers[field] = transformer

    def encode_as_df(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes the passed DataFrame into values between 0 and 1.
        Only encodes columns this DataEncoder knows about, i.e. columns that were in the DataFrame this DataEncoder
        was instantiated with
        :param data_df: the DataFrame to encode
        :return: a new, encoded DataFrame
        """
        values_by_column = {}
        for column in data_df.columns:
            # UN-736: Only encode columns that have a transformer. Ignore the others.
            if column in self.transformers:
                field_type = self.fields[column]["data_type"]
                field_dtype = UNILEAF_TO_PANDAS_TYPE[field_type]
                transformer = self.transformers[column]
                raw_values = data_df[[column]].astype(field_dtype)
                is_label = isinstance(transformer, LabelEncoder)
                is_categorical = self._is_categorical(column)

                if is_label:
                    # LabelEncoder wants a row instead of column
                    raw_values = np.ravel(raw_values, order='C')

                # Encode
                encoded_values = transformer.transform(raw_values)

                if is_categorical:
                    # Categorical value, encoded as a numpy array.
                    # Convert it to list for Pandas to handle it nicely
                    # Source:
                    # https://stackoverflow.com/questions/45548426/store-numpy-array-in-cells-of-a-pandas-dataframe
                    values_by_column[column] = encoded_values.tolist()
                else:
                    # Numerical value, encoded as scalars.
                    values_by_column[column] = encoded_values.squeeze().tolist()
        encoded_df = pd.DataFrame.from_records(values_by_column,
                                               index=list(range(data_df.shape[0]))
                                               )[values_by_column.keys()]
        return encoded_df

    def decode_as_df(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Decodes the passed DataFrame back into original value ranges
        :param data_df: the DataFrame to decode
        :return: a new, decoded DataFrame
        """
        values_by_column = {}
        for column in data_df.columns:
            transformer = self.transformers[column]
            field_type = self.fields[column]["data_type"]
            field_dtype = UNILEAF_TO_PANDAS_TYPE[field_type]
            is_categorical = self._is_categorical(column)

            if is_categorical:
                # This is a one hot vector column
                raw_values = data_df[column].to_list()
            else:
                # This is a numerical column
                raw_values = data_df[[column]]
                # UN-1060 If values are in a list of list, convert them to a single list
                if data_df[column].dtype == object:
                    raw_values = data_df[column].to_list()
                # Clip values to make sure they are within the encoder's range before decoding
                raw_values = np.clip(raw_values, 0, 1)
                if self._is_outcome(column):
                    # Do not round outcomes: keep as float. Predictions for binary outcomes can be probabilities
                    field_dtype = float

            # Convert values back to original scale
            column_values = transformer.inverse_transform(raw_values)
            if field_dtype == int:
                # Round values to nearest int instead of truncating
                column_values = np.rint(column_values)

            values_by_column[column] = np.array(column_values.ravel(order='C'), dtype=field_dtype)
        decoded_df = pd.DataFrame.from_records(values_by_column,
                                               index=list(range(data_df.shape[0]))
                                               )[values_by_column.keys()]
        return decoded_df

    @staticmethod
    def encoded_df_to_np(data_df: pd.DataFrame) -> np.ndarray:
        """
        Converts an encoded Pandas DataFrame into a numpy array
        :param data_df: a DataFrame containing encoded data, for instance rows like [1, 0, 0], 0.9, [1,0]
        :return: a numpy array of scalar features, for instance rows like [1, 0, 0, 0.9, 1, 0]
        """
        concatenated_rows = []
        nb_rows = len(data_df)
        data_array = data_df.to_numpy()
        for row in range(nb_rows):
            row_array = data_array[row]
            # Concatenate each array (categorical value) and scalar (numerical value) into a single array
            concatenated_rows.append(np.hstack(row_array))
        features = np.array(concatenated_rows)
        return features

    @staticmethod
    def np_to_encoded_df(np_features: np.ndarray,
                         column_length: Dict) -> pd.DataFrame:
        """
        Converts an encoded numpy array into a DataFrame with columns containing the right number of values
        :param np_features: a numpy array
        :param column_length: dictionary of column name to column length
        :return: a Pandas DataFrame
        """
        # The numpy array may contain more than one column. Split it into chunks corresponding to each column.
        i = 0
        indices = []
        for value in column_length.values():
            i += value
            indices.append(i)

        # Split the array into chunks matching the number of values expected for each column
        if len(column_length) > 1:
            cols = np.split(np_features, indices, axis=1)
        else:
            # Single column, nothing to split
            cols = [np_features]

        features = {}
        for i, column in enumerate(column_length.keys()):
            if column_length[column] > 1:
                # Categorical one-hots: keep the array as the value of the column, e.g. [1, 0, 0]
                features[column] = list(cols[i])
            else:
                # Numerical, 1 scalar: flatten the array
                features[column] = cols[i].ravel(order='C')
        encoded_df = pd.DataFrame(features)
        return encoded_df

    def _is_categorical(self, field_name: str) -> bool:
        """
        Returns True if this field is defined as CATEGORICAL in the fields definition
        :param field_name: The name of the field to check
        :return: True if the field is categorical, False otherwise
        """
        is_categorical = self.fields[field_name]["valued"].lower() == "categorical"
        return is_categorical

    def _is_numerical(self, field_name: str) -> bool:
        """
        Returns True if this field is defined as numerical, i.e. either a FLOAT or an INT,  in the fields definition
        :param field_name: The name of the field to check
        :return: True if the field is a feature, False otherwise
        """
        data_type = self.fields[field_name]["data_type"].lower()
        return data_type in ["int", "float"]

    def _is_feature(self, field_name: str) -> bool:
        """
        Returns True if this field is defined as a feature, i.e. either a CONTEXT or an ACTION, in the fields definition
        :param field_name: The name of the field to check
        :return: True if the field is a feature, False otherwise
        """
        empty = []
        context = self.cao_mapping.get("context", empty)
        actions = self.cao_mapping.get("actions", empty)
        is_feature = field_name in context or field_name in actions
        return is_feature

    def _is_outcome(self, field_name: str) -> bool:
        """
        Returns True if this field is defined as an OUTCOME in the fields definition
        :param field_name: The name of the field to check
        :return: True if the field is an OUTCOME, False otherwise
        """
        empty = []
        outcomes = self.cao_mapping.get("outcomes", empty)
        is_outcome = field_name in outcomes
        return is_outcome
