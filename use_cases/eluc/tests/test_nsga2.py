"""
Unit tests for the NSGA-II Torch implementation.
"""

import unittest

import pandas as pd
import torch

from data import constants
from data.eluc_data import ELUCData
from predictors.sklearn.sklearn_predictor import LinearRegressionPredictor
from prescriptors.nsga2.candidate import Candidate
from prescriptors.nsga2.torch_prescriptor import TorchPrescriptor

class TestTorchPrescriptor(unittest.TestCase):
    """
    Tests PyTorch prescriptor class
    """

    @classmethod
    def setUpClass(cls):
        predictor = LinearRegressionPredictor(features=constants.DIFF_LAND_USE_COLS, n_jobs=-1)
        predictor.load("predictors/sklearn/trained_models/no_overlap_linreg")
        cls.dataset = ELUCData(start_year=2020, test_year=2021, end_year=2022, countries=["US"])
        cls.prescriptor = TorchPrescriptor(
            100,
            100,
            0.2,
            cls.dataset.train_df,
            cls.dataset.encoder,
            predictor,
            2048,
            {"in_size": len(constants.CAO_MAPPING["context"]), "hidden_size": 16, "out_size": len(constants.RECO_COLS)}
        )

    def test_reco_tensor_to_df(self):
        """
        Takes a tensor of recommendations and converts it to a scaled DataFrame.
        Makes sure the scaled dataframe's rows sum to the original sum of land use and
        that the index is the same.
        """
        reco_tensor = torch.rand(100, len(constants.RECO_COLS))
        context_df = self.prescriptor.eval_df.iloc[:100]
        reco_df = self.prescriptor._reco_tensor_to_df(reco_tensor, context_df)
        self.assertIsInstance(reco_df, pd.DataFrame)
        self.assertEqual(reco_df.shape, (100, len(constants.RECO_COLS)))
        self.assertEqual(reco_df.sum(axis=1).all(), context_df[constants.RECO_COLS].sum(axis=1).all())
        self.assertTrue(reco_df.index.equals(context_df.index))

    def test_reco_tensor_to_df_all_zero_tensor(self):
        """
        Tests the case where the tensor is all zeros.
        """
        reco_tensor = torch.zeros(100, len(constants.RECO_COLS))
        context_df = self.prescriptor.eval_df.iloc[:100]
        reco_df = self.prescriptor._reco_tensor_to_df(reco_tensor, context_df)
        self.assertIsInstance(reco_df, pd.DataFrame)
        self.assertEqual(reco_df.shape, (100, len(constants.RECO_COLS)))
        self.assertEqual(reco_df.sum(axis=1).all(), context_df[constants.RECO_COLS].sum(axis=1).all())
        self.assertTrue(reco_df.index.equals(context_df.index))

    def test_reco_tensor_to_df_all_zero_context(self):
        """
        Tests the case where the context dataframe is all zeros.
        """
        reco_tensor = torch.rand(100, len(constants.RECO_COLS))
        context_df = pd.DataFrame(0, index=range(0, 200, 2), columns=constants.RECO_COLS)
        reco_df = self.prescriptor._reco_tensor_to_df(reco_tensor, context_df)
        self.assertIsInstance(reco_df, pd.DataFrame)
        self.assertEqual(reco_df.shape, (100, len(constants.RECO_COLS)))
        self.assertEqual(reco_df.sum(axis=1).all(), context_df[constants.RECO_COLS].sum(axis=1).all())
        self.assertTrue(reco_df.index.equals(context_df.index))

    def test_reco_to_context_actions(self):
        """
        Tests the conversion of a recommendation df and a context df to a context actions df.
        Makes sure the difference between the context and recommendation is what is output.
        Also makes sure diff for all the NO_CHANGE_COLS is 0.
        TODO: This isn't a great test - should I redo it with synthetic data?
        """
        reco_df = self.prescriptor.eval_df.iloc[:100][constants.RECO_COLS]
        self.assertTrue(reco_df.sum(axis=1).all() > 0)
        self.assertTrue(reco_df.sum(axis=1).all() <= 1)

        context_df = self.prescriptor.eval_df.iloc[100:200][constants.CAO_MAPPING["context"]].copy()
        self.assertTrue(context_df.sum(axis=1).all() > 0)
        self.assertTrue(context_df.sum(axis=1).all() <= 1)
        # This has to be true - fudge it
        context_df = context_df.set_index(reco_df.index)

        context_actions_df = self.prescriptor._reco_to_context_actions(reco_df, context_df)

        diff_df = reco_df - context_df[constants.RECO_COLS]
        diff_df = diff_df.rename(constants.RECO_MAP, axis=1)
        diff_df[constants.NO_CHANGE_COLS] = 0
        diff_df = diff_df[constants.DIFF_LAND_USE_COLS]

        self.assertTrue((diff_df == context_actions_df[constants.DIFF_LAND_USE_COLS]).all().all())

    def test_compute_percent_changed_indices_same(self):
        """
        Makes sure the indices in change_df are the same as in the context_actions_df.
        """
        context_actions_df = self.prescriptor.eval_df.iloc[:100]
        context_actions_df = context_actions_df[constants.CAO_MAPPING["context"] + constants.CAO_MAPPING["actions"]]
        change_df = self.prescriptor._compute_percent_changed(context_actions_df)

        self.assertTrue(change_df.index.equals(context_actions_df.index))

    def test_prescribe_indices_same(self):
        """
        Tests prescribe method to see if context_actions_df has the same indices as the input context_df.
        """
        candidate = Candidate(in_size=len(constants.CAO_MAPPING["context"]),
                              hidden_size=16,
                              out_size=len(constants.RECO_COLS))
        context_df = self.dataset.test_df.iloc[:100][constants.CAO_MAPPING["context"]]
        context_actions_df = self.prescriptor._prescribe(candidate, context_df)

        self.assertTrue(context_actions_df.index.equals(context_df.index))
        self.assertTrue(context_df.equals(context_actions_df[constants.CAO_MAPPING["context"]]))

if __name__ == "__main__":
    unittest.main()
