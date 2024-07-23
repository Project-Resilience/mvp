"""
Tests prescriptor LandUsePrescriptor class.
"""
from pathlib import Path
import shutil
import unittest

import pandas as pd
import torch

from data import constants
from data.eluc_encoder import ELUCEncoder
from persistence.serializers.prescriptor_serializer import PrescriptorSerializer
from prescriptors.nsga2.candidate import Candidate
from prescriptors.nsga2.land_use_prescriptor import LandUsePrescriptor


class TestLandUsePrescriptor(unittest.TestCase):
    """
    Tests PyTorch prescriptor class
    """
    @classmethod
    def setUpClass(cls):
        """
        Set up tests by reading dummy data from csv in repo.
        """
        test_df = pd.read_csv(Path("tests/test_data.csv"))
        test_df["time_idx"] = test_df["time"]
        test_df["lat_idx"] = test_df["lat"]
        test_df["lon_idx"] = test_df["lon"]
        test_df = test_df.set_index(["time_idx", "lat_idx", "lon_idx"], drop=True)
        cls.df = test_df

        encoder = ELUCEncoder.from_pandas(test_df)

        candidate = Candidate(len(constants.CAO_MAPPING["context"]), 16, len(constants.RECO_COLS))
        cls.prescriptor = LandUsePrescriptor(candidate, encoder)

        # For serialization tests
        cls.temp_path = Path("tests/temp")

    # Disable protected access warning so we can test the private methods
    # pylint: disable=protected-access
    def test_reco_tensor_to_df_all_zero_tensor(self):
        """
        Tests the case where the tensor is all zeros.
        """
        reco_tensor = torch.zeros(len(self.df), len(constants.RECO_COLS))
        context_df = self.df[constants.CAO_MAPPING["context"]]
        reco_df = self.prescriptor._reco_tensor_to_df(reco_tensor, context_df)
        self.assertIsInstance(reco_df, pd.DataFrame)
        self.assertEqual(reco_df.shape, (len(self.df), len(constants.RECO_COLS)))
        self.assertEqual(reco_df.sum(axis=1).all(), context_df[constants.RECO_COLS].sum(axis=1).all())
        self.assertTrue(reco_df.index.equals(context_df.index))

    def test_reco_tensor_to_df_all_zero_context(self):
        """
        Tests the case where the context dataframe is all zeros.
        """
        reco_tensor = torch.rand(10, len(constants.RECO_COLS))
        zero_df = self.df.copy()
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
        reco_df = self.df[constants.RECO_COLS]
        self.assertTrue(reco_df.sum(axis=1).all() > 0)
        self.assertTrue(reco_df.sum(axis=1).all() <= 1)

        context_df = self.df[constants.CAO_MAPPING["context"]].copy()
        self.assertTrue(context_df.sum(axis=1).all() > 0)
        self.assertTrue(context_df.sum(axis=1).all() <= 1)

        context_actions_df = self.prescriptor._reco_to_context_actions(reco_df, context_df)

        diff_df = reco_df - context_df[constants.RECO_COLS]
        diff_df = diff_df.rename(constants.RECO_MAP, axis=1)
        diff_df[constants.NO_CHANGE_COLS] = 0
        diff_df = diff_df[constants.DIFF_LAND_USE_COLS]

        self.assertTrue((diff_df == context_actions_df[constants.DIFF_LAND_USE_COLS]).all().all())
    # pylint: enable=protected-access

    def test_prescribe_indices_same(self):
        """
        Tests prescribe method to see if context_actions_df has the same indices as the input context_df.
        """
        context_df = self.df[constants.CAO_MAPPING["context"]]
        context_actions_df = self.prescriptor.prescribe(context_df)

        self.assertTrue(context_actions_df.index.equals(context_df.index))
        self.assertTrue(context_df.equals(context_actions_df[constants.CAO_MAPPING["context"]]))

    def test_serialization(self):
        """
        Makes sure after we save to disk and load from disk, the prescriptor is the same.
        """
        persistor = PrescriptorSerializer()
        persistor.save(self.prescriptor, self.temp_path / "prescriptor")

        loaded_prescriptor = persistor.load(self.temp_path / "prescriptor")

        for old_param, new_param in zip(self.prescriptor.candidate.parameters(),
                                        loaded_prescriptor.candidate.parameters()):

            self.assertTrue(torch.equal(old_param, new_param))

        self.assertEqual(self.prescriptor.encoder.fields, loaded_prescriptor.encoder.fields)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up temp directory.
        """
        shutil.rmtree(cls.temp_path)
