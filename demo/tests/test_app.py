import unittest
import pandas as pd
import json

import app.app as app
import app.constants as constants
import app.utils as utils


class TestUtilFunctions(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv(constants.DATA_FILE_PATH, index_col=constants.INDEX_COLS)

    def test_add_nonland(self):
        """
        Simple vanilla test case for add_nonland().
        """
        data = [0, 0.01, 0.01, 0.2, 0.4, 0.02, 0.03, 0.01, 0.01, 0.05, 0.01, 0.1]
        series = pd.Series(dict(zip(constants.LAND_USE_COLS, data)))
        full = utils.add_nonland(series)
        self.assertAlmostEqual(full["nonland"], 1 - sum(data), delta=constants.SLIDER_PRECISION)

    def test_add_nonland_sum_over_one(self):
        """
        Makes sure if the columns sum to >1, we get 0 for nonland
        """
        data = [1 for _ in range(len(constants.LAND_USE_COLS))]
        series = pd.Series(dict(zip(constants.LAND_USE_COLS, data)))
        full = utils.add_nonland(series)
        self.assertAlmostEqual(full["nonland"], 0, delta=constants.SLIDER_PRECISION)

    def test_create_check_options_length(self):
        values = ["a", "b", "c"]
        options = utils.create_check_options(values)
        self.assertEqual(len(options), len(values))

    def test_create_check_options_values(self):
        """
        Checks if the values in the options are correct
        """
        values = ["a", "b", "c"]
        options = utils.create_check_options(values)
        for i in range(len(options)):
            self.assertEqual(options[i]["value"], values[i])

    def test_compute_percent_change(self):
        """
        Tests compute percent change on standard example.
        """
        context_data = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12]
        presc_data = [0.10, 0.06, 0.11, 0.05, 0.12, 0.04, 0.13, 0.03, 0.08]
        context = pd.Series(dict(zip(constants.LAND_USE_COLS, context_data)))
        presc = pd.Series(dict(zip(constants.RECO_COLS, presc_data)))

        percent_change = utils.compute_percent_change(context, presc)
        self.assertAlmostEqual(percent_change, 0.14, delta=constants.SLIDER_PRECISION)

    def test_compute_percent_change_no_change(self):
        """
        Tests compute percent change when nothing changes.
        """
        context_data = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12]
        presc_data = context_data[0:6] + context_data [8:11]
        context = pd.Series(dict(zip(constants.LAND_USE_COLS, context_data)))
        presc = pd.Series(dict(zip(constants.RECO_COLS, presc_data)))

        percent_change = utils.compute_percent_change(context, presc)
        self.assertAlmostEqual(percent_change, 0, delta=constants.SLIDER_PRECISION)

    def test_compute_percent_change_all_nonreco(self):
        """
        Tests compute change when there is only urban/primf/primn.
        """
        context_data = [0, 0, 0, 0, 0, 0, 0.33, 0.33, 0, 0, 0, 0.34]
        presc_data = context_data[0:6] + context_data [8:11]
        context = pd.Series(dict(zip(constants.LAND_USE_COLS, context_data)))
        presc = pd.Series(dict(zip(constants.RECO_COLS, presc_data)))

        percent_change = utils.compute_percent_change(context, presc)
        self.assertEqual(percent_change, 0)

    def test_compute_percent_change_not_sum_to_one(self):
        """
        Tests compute percent change on a context with some nonland.
        """
        context_data = [0.01 for _ in range(len(constants.LAND_USE_COLS))]
        presc_data = [0.02, 0.00, 0.02, 0.00, 0.02, 0.00, 0.02, 0.00, 0.01]
        context = pd.Series(dict(zip(constants.LAND_USE_COLS, context_data)))
        presc = pd.Series(dict(zip(constants.RECO_COLS, presc_data)))

        percent_change = utils.compute_percent_change(context, presc)
        self.assertAlmostEqual(percent_change, 0.333333, delta=constants.SLIDER_PRECISION)


class TestEncoder(unittest.TestCase):
    """
    Since the encoded values are somewhat arbitrary based off what the prescriptor
    is trained on, we have to test based off what is in the fields file.
    """

    def setUp(self):
        self.df = pd.read_csv(constants.DATA_FILE_PATH, index_col=constants.INDEX_COLS)
        self.encoder = None
        self.fields = None
        with open(constants.FIELDS_PATH, "r") as f:
            self.fields = json.load(f)
            self.encoder = utils.Encoder(self.fields)

    def test_easy_case(self):
        """
        Tests encoding a simple case.
        """
        row = self.df.iloc[[0]]
        row = row[constants.CONTEXT_COLUMNS]
        pred = self.encoder.encode_as_df(row)

        for col in constants.CONTEXT_COLUMNS:
            range = self.fields[col]["range"]
            # Min-max scale formula
            true = (row[col].values[0] - range[0]) / (range[1] - range[0])
            self.assertAlmostEqual(pred[col].values[0], true, delta=constants.SLIDER_PRECISION)

    def test_non_field_cols(self):
        """
        Test that non-field columns are not encoded and excluded from final dataframe.
        """
        row = self.df.iloc[[0]]
        row = row[constants.CONTEXT_COLUMNS]
        row["test"] = 999
        enc = self.encoder.encode_as_df(row)
        # Make sure we didn't add the test column
        self.assertEqual(sorted(list(enc.columns)), sorted(constants.CONTEXT_COLUMNS))

        # Make sure we're still encoding
        true = (row["primf"].values[0] - self.fields["primf"]["range"][0]) / (self.fields["primf"]["range"][1] - self.fields["primf"]["range"][0])
        self.assertAlmostEqual(enc["primf"].values[0], true, delta=constants.SLIDER_PRECISION)

    def test_multiple_input(self):
        """
        Tests we can pass in a multi-row dataframe and get proper encodings.
        This isn't strictly necessary for our current use case, but it's good to test.
        """
        rows = self.df.iloc[0:2]
        rows = rows[constants.CONTEXT_COLUMNS]
        enc = self.encoder.encode_as_df(rows)

        for col in constants.CONTEXT_COLUMNS:
            minmax = self.fields[col]["range"]
            for i in range(len(rows)):
                val = rows.iloc[i][col]
                true = (val - minmax[0]) / (minmax[1] - minmax[0])
                self.assertAlmostEqual(enc.iloc[i][col], true, delta=constants.SLIDER_PRECISION)