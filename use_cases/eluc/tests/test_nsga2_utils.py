"""
Unit tests for the NSGA-II Torch implementation.
"""
import unittest

import numpy as np

from prescriptors.nsga2.candidate import Candidate
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
