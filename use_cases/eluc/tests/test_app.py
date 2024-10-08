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
        df = pd.DataFrame([data], columns=constants.LAND_USE_COLS)
        full = utils.add_nonland(df)
        self.assertAlmostEqual(full["nonland"].iloc[0], 1 - sum(data), delta=app_constants.SLIDER_PRECISION)

    def test_add_nonland_sum_over_one(self):
        """
        Makes sure if the columns sum to >1, we get 0 for nonland
        """
        data = [1 for _ in range(len(constants.LAND_USE_COLS))]
        df = pd.DataFrame([data], columns=constants.LAND_USE_COLS)
        full = utils.add_nonland(df)
        self.assertAlmostEqual(full["nonland"].iloc[0], 0, delta=app_constants.SLIDER_PRECISION)
