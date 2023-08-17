import unittest
import pandas as pd

import app.app as app
import app.constants as constants
import app.utils as utils

class TestUtils(unittest.TestCase):

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

    # TODO: Why doesn't this work with idx=0?
    def test_create_map(self):
        """
        Checks if created map has correct point colored red.
        """
        idx = 1
        present = self.df.loc[2021].copy().reset_index()
        lat = present.iloc[idx]["lat"]
        lon = present.iloc[idx]["lon"]
        fig = utils.create_map(
            present, 
            constants.MAP_COORDINATE_DICT["UK"]["lat"], 
            constants.MAP_COORDINATE_DICT["UK"]["lon"], 
            zoom=constants.MAP_COORDINATE_DICT["UK"]["zoom"], 
            color_idx=idx
        )
        customdata = fig['data'][1]['customdata'][0]
        self.assertEqual(customdata[0], lat)
        self.assertEqual(customdata[1], lon)
        self.assertEqual(customdata[2], "red")

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
        context = pd.Series(dict(zip(constants.LAND_USE_COLS, context_data)))
        presc = pd.Series(dict(zip(constants.RECO_COLS, context_data)))

        percent_change = utils.compute_percent_change(context, presc)
        self.assertAlmostEqual(percent_change, 0, delta=constants.SLIDER_PRECISION)

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