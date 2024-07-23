"""
Unit tests for the NSGA-II Torch implementation.
"""
import itertools
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

    def manual_two_obj_dominate(self, candidate1: Candidate, candidate2: Candidate) -> bool:
        """
        Manually test all cases of domination for 2 objectives.
        For candidate 1 to dominate candidate 2 it must:
            - Have a lower value in at least one objective
            - Have lower or equal values in all the rest
        """
        if (candidate1.metrics[0] < candidate2.metrics[0]) and (candidate1.metrics[1] <= candidate2.metrics[1]):
            return True
        if (candidate1.metrics[0] <= candidate2.metrics[0]) and (candidate1.metrics[1] < candidate2.metrics[1]):
            return True
        return False

    def test_domination_two_obj(self):
        """
        Tests domination works in all possible cases.
        Get all combinations of pairs of values [0, 1, 2] for each objective and tests against the manual checker.
        """
        for comb in itertools.combinations([0, 1, 2], 4):
            candidate1 = Candidate(16, 16, 16)
            candidate2 = Candidate(16, 16, 16)
            candidate1.metrics = [comb[0], comb[1]]
            candidate2.metrics = [comb[2], comb[3]]
            self.assertEqual(nsga2_utils.dominates(candidate1, candidate2),
                             self.manual_two_obj_dominate(candidate1, candidate2))
