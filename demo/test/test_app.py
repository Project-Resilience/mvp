import unittest
import pandas as pd

import app.app as app
import app.constants as constants
import app.utils as utils

class TestUtils(unittest.TestCase):
    # def setUp(self):
    #     self.df = pd.read_csv(constants.DATA_FILE_PATH, index_col=constants.INDEX_COLS)

    def test_add_nonland(self):
        data = [0, 0.01, 0.01, 0.2, 0.4, 0.02, 0.03, 0.01, 0.01, 0.05, 0.01, 0.1]
        series = pd.Series(dict(zip(constants.LAND_USE_COLS, data)))
        full = utils.add_nonland(series)
        self.assertAlmostEqual(full["nonland"], 1 - sum(data), delta=constants.SLIDER_PRECISION)