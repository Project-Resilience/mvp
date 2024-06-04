"""
Unit tests for the NSGA-II Torch implementation.
"""
import unittest

import numpy as np
import pandas as pd
import torch

from data import constants
from data.eluc_data import ELUCData
from data.eluc_data import ELUCEncoder
from prescriptors.nsga2.candidate import Candidate
from prescriptors.nsga2.land_use_prescriptor import LandUsePrescriptor
from prescriptors.nsga2 import nsga2_utils

class TestNSGA2Utils(unittest.TestCase):
    """
    Tests the NGSA-II utility functions.
    """
    def test_distance_calculation(self):
        """
        Tests the calculation of crowding distance.
        Objective 1 is the candidate number times 2.
        Objective 2 is the candidate number squared.
        """
        # Create a dummy front
        front = []
        tgt_distances = []
        for i in range(4):
            dummy_candidate = Candidate(16, 16, 16)
            dummy_candidate.metrics = [i*2, i**2]
            front.append(dummy_candidate)
            if i in {0, 3}:
                tgt_distances.append(np.inf)
            else:
                dist0 = ((i + 1) * 2 - (i - 1) * 2) / 6
                dist1 = ((i + 1)**2 - (i - 1)**2) / 9
                tgt_distances.append(dist0 + dist1)

        # Manually shuffle the front
        shuffled_indices = [1, 3, 0, 2]
        shuffled_front = [front[i] for i in shuffled_indices]
        shuffled_tgts = [tgt_distances[i] for i in shuffled_indices]

        # Assign crowding distances
        nsga2_utils.calculate_crowding_distance(shuffled_front)
        for candidate, tgt in zip(shuffled_front, shuffled_tgts):
            self.assertAlmostEqual(candidate.distance, tgt)

class TestLandUsePrescriptor(unittest.TestCase):
    """
    Tests PyTorch prescriptor class
    """
    @classmethod
    def setUpClass(cls):
        data = ELUCData()
        cls.df = data.train_df

        candidate = Candidate(len(constants.CAO_MAPPING["context"]), 16, len(constants.RECO_COLS))
        cls.prescriptor = LandUsePrescriptor(candidate, data.encoder)

        cls.n = 10

    def test_reco_tensor_to_df_all_zero_tensor(self):
        """
        Tests the case where the tensor is all zeros.
        """
        reco_tensor = torch.zeros(self.n, len(constants.RECO_COLS))
        context_df = self.df[constants.CAO_MAPPING["context"]].iloc[:self.n]
        reco_df = self.prescriptor._reco_tensor_to_df(reco_tensor, context_df)
        self.assertIsInstance(reco_df, pd.DataFrame)
        self.assertEqual(reco_df.shape, (self.n, len(constants.RECO_COLS)))
        self.assertEqual(reco_df.sum(axis=1).all(), context_df[constants.RECO_COLS].sum(axis=1).all())
        self.assertTrue(reco_df.index.equals(context_df.index))

    def test_reco_tensor_to_df_all_zero_context(self):
        """
        Tests the case where the context dataframe is all zeros.
        """
        reco_tensor = torch.rand(10, len(constants.RECO_COLS))
        zero_df = self.df.iloc[:self.n].copy()
        zero_df[constants.LAND_USE_COLS] = 0
        reco_df = self.prescriptor._reco_tensor_to_df(reco_tensor, zero_df)
        self.assertIsInstance(reco_df, pd.DataFrame)
        self.assertEqual(reco_df.shape, (10, len(constants.RECO_COLS)))
        self.assertEqual(reco_df.sum(axis=1).all(), zero_df[constants.RECO_COLS].sum(axis=1).all())
        self.assertTrue(reco_df.index.equals(zero_df.index))

    def test_reco_to_context_actions(self):
        """
        Tests the conversion of a recommendation df and a context df to a context actions df.
        Makes sure the difference between the context and recommendation is what is output.
        Also makes sure diff for all the NO_CHANGE_COLS is 0.
        TODO: This isn't a great test - should I redo it with synthetic data?
        """
        reco_df = self.df.iloc[:self.n][constants.RECO_COLS]
        self.assertTrue(reco_df.sum(axis=1).all() > 0)
        self.assertTrue(reco_df.sum(axis=1).all() <= 1)

        context_df = self.df.iloc[:self.n][constants.CAO_MAPPING["context"]].copy()
        self.assertTrue(context_df.sum(axis=1).all() > 0)
        self.assertTrue(context_df.sum(axis=1).all() <= 1)

        context_actions_df = self.prescriptor._reco_to_context_actions(reco_df, context_df)

        diff_df = reco_df - context_df[constants.RECO_COLS]
        diff_df = diff_df.rename(constants.RECO_MAP, axis=1)
        diff_df[constants.NO_CHANGE_COLS] = 0
        diff_df = diff_df[constants.DIFF_LAND_USE_COLS]

        self.assertTrue((diff_df == context_actions_df[constants.DIFF_LAND_USE_COLS]).all().all())

    def test_prescribe_indices_same(self):
        """
        Tests prescribe method to see if context_actions_df has the same indices as the input context_df.
        """
        context_df = self.df.iloc[:self.n][constants.CAO_MAPPING["context"]]
        context_actions_df = self.prescriptor.prescribe(context_df)

        self.assertTrue(context_actions_df.index.equals(context_df.index))
        self.assertTrue(context_df.equals(context_actions_df[constants.CAO_MAPPING["context"]]))

if __name__ == "__main__":
    unittest.main()
