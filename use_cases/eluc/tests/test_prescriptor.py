"""
Tests the generic prescriptor class.
TODO: This will eventually have to be moved into a ProjectResilienceSDK.
"""
import unittest

import pandas as pd

from data import constants
from prescriptors.prescriptor_manager import PrescriptorManager

class TestComputeChange(unittest.TestCase):
    """
    Tests the prescriptor compute change method.
    """
    def setUp(self):
        """
        Sets up a dummy prescriptor. This doesn't even really have to be instantiated properly it just needs to
        be able to call compute_percent_changed.
        """
        self.prescriptor_manager = PrescriptorManager(None, None)

    def _list_data_to_df(self, context_data: list, presc_data: list) -> pd.DataFrame:
        """
        Helper function that takes a list of context and prescribed data and converts it to a context actions df
        to be used by the prescriptor.
        context_data should be length constants.LAND_USE_COLS
        presc_data should be length constants.RECO_COLS
        """
        context_actions_df = pd.DataFrame([dict(zip(constants.LAND_USE_COLS, context_data))])
        presc = pd.DataFrame([dict(zip(constants.RECO_COLS, presc_data))])

        diff = presc[constants.RECO_COLS] - context_actions_df[constants.RECO_COLS]
        diff.rename(columns=constants.RECO_MAP, inplace=True)
        context_actions_df[constants.DIFF_RECO_COLS] = diff[constants.DIFF_RECO_COLS]

        context_actions_df[constants.NO_CHANGE_COLS] = 0
        context_actions_df[constants.NONLAND_FEATURES] = 0

        return context_actions_df

    def test_compute_percent_change(self):
        """
        Tests compute percent change on standard example.
        """
        even_amt = 1 / len(constants.LAND_USE_COLS)
        context_data = [even_amt for _ in range(len(constants.LAND_USE_COLS))]
        presc_data = [even_amt * 2, 0, even_amt * 2, 0, even_amt]

        context_actions_df = self._list_data_to_df(context_data, presc_data)

        percent_change = self.prescriptor_manager.compute_percent_changed(context_actions_df)["change"].iloc[0]
        self.assertAlmostEqual(percent_change, even_amt * 2)

    def test_compute_percent_change_no_change(self):
        """
        Tests compute percent change when nothing changes.
        """
        context_data = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12]
        presc_data = context_data[0:6] + context_data [8:11]

        context_actions_df = self._list_data_to_df(context_data, presc_data)

        percent_change = self.prescriptor_manager.compute_percent_changed(context_actions_df)["change"].iloc[0]
        self.assertAlmostEqual(percent_change, 0)

    def test_compute_percent_change_all_nonreco(self):
        """
        Tests compute change when there is only urban/primf/primn.
        """
        context_data = [0, 0, 0, 0, 0, 0, 0.33, 0.33, 0, 0, 0, 0.34]
        presc_data = context_data[0:6] + context_data [8:11]

        context_actions_df = self._list_data_to_df(context_data, presc_data)

        percent_change = self.prescriptor_manager.compute_percent_changed(context_actions_df)["change"].iloc[0]
        self.assertEqual(percent_change, 0)

    def test_compute_percent_change_not_sum_to_one(self):
        """
        Tests compute percent change on a context with some nonland.
        """
        context_data = [0.01 for _ in range(len(constants.LAND_USE_COLS))]
        presc_data = [0.02, 0.00, 0.02, 0.00, 0.01]

        context_actions_df = self._list_data_to_df(context_data, presc_data)

        percent_change = self.prescriptor_manager.compute_percent_changed(context_actions_df)["change"].iloc[0]

        self.assertAlmostEqual(percent_change, 0.02 / (0.01 * len(constants.LAND_USE_COLS)))

    def test_compute_percent_changed_indices_same(self):
        """
        Makes sure the indices in change_df are the same as in the context_actions_df.
        """
        context_data = [0.01 for _ in range(len(constants.LAND_USE_COLS))]
        presc_data = [0.02, 0.00, 0.02, 0.00, 0.01]

        context_actions_df = self._list_data_to_df(context_data, presc_data)
        change_df = self.prescriptor_manager.compute_percent_changed(context_actions_df)

        self.assertTrue(change_df.index.equals(context_actions_df.index))
