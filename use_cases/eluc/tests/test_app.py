"""
Tests the demo app
"""
import unittest
import pandas as pd

import app.constants as app_constants
from app import utils
from data import constants

class TestUtilFunctions(unittest.TestCase):
    """
    Tests app utilities.
    """
    def test_add_nonland(self):
        """
        Simple vanilla test case for add_nonland(). Makes sure the nonland column added equals whatever we need left.
        """
        data = [0, 0.01, 0.01, 0.2, 0.4, 0.02, 0.03, 0.01]
        series = pd.Series(dict(zip(constants.LAND_USE_COLS, data)))
        full = utils.add_nonland(series)
        self.assertAlmostEqual(full["nonland"], 1 - sum(data), delta=app_constants.SLIDER_PRECISION)

    def test_add_nonland_sum_over_one(self):
        """
        Makes sure if the columns sum to >1, we get 0 for nonland
        """
        data = [1 for _ in range(len(constants.LAND_USE_COLS))]
        series = pd.Series(dict(zip(constants.LAND_USE_COLS, data)))
        full = utils.add_nonland(series)
        self.assertAlmostEqual(full["nonland"], 0, delta=app_constants.SLIDER_PRECISION)

    def test_create_check_options_length(self):
        """
        Makes sure the checklist we create has the same number of options as the input.
        """
        values = ["a", "b", "c"]
        options = utils.create_check_options(values)
        self.assertEqual(len(options), len(values))

    def test_create_check_options_values(self):
        """
        Checks if the values in the options are correct
        """
        values = ["a", "b", "c"]
        options = utils.create_check_options(values)
        for i, option in enumerate(options):
            self.assertEqual(option["value"], values[i])
