"""Held-out regret and win rate (architecture.md §7, §8)."""

import unittest

import numpy as np

from data.contract import Lineage, Version
from data.simulator import simulate_lineages
from metrics import direction_agreement, held_out_regret, oracle_best_index, win_rate

NO_EDIT = {"edit_files": 0, "edit_lines": 0, "touched_verification": False, "revision_calls": 0}


def toy_lineage(feedback, heldout):
    versions = tuple(Version(float(f), float(h), dict(NO_EDIT)) for f, h in zip(feedback, heldout))
    return Lineage("toy", versions)


class RegretTest(unittest.TestCase):
    def test_zero_regret_when_oracle_best_is_chosen(self):
        for lineage in simulate_lineages(seed=5):
            self.assertEqual(held_out_regret(lineage, oracle_best_index(lineage)), 0.0)

    def test_regret_is_gap_to_heldout_best(self):
        lineage = toy_lineage(feedback=[50, 60, 55], heldout=[52, 48, 57])
        self.assertEqual(oracle_best_index(lineage), 2)
        self.assertAlmostEqual(held_out_regret(lineage, 1), 9.0)
        self.assertAlmostEqual(held_out_regret(lineage, 0), 5.0)

    def test_regret_is_never_negative(self):
        for lineage in simulate_lineages(seed=6):
            for i in range(len(lineage)):
                self.assertGreaterEqual(held_out_regret(lineage, i), 0.0)

    def test_out_of_range_index_raises(self):
        with self.assertRaises(IndexError):
            held_out_regret(toy_lineage([1, 2], [1, 2]), 2)


class WinRateTest(unittest.TestCase):
    def test_counts_strict_wins_only(self):
        rule = np.array([0.0, 1.0, 2.0, 3.0])
        baseline = np.array([1.0, 1.0, 1.0, 4.0])
        self.assertAlmostEqual(win_rate(rule, baseline), 0.5)  # ties are not wins

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            win_rate(np.zeros(3), np.zeros(4))


class DirectionAgreementTest(unittest.TestCase):
    def test_counts_same_sign_moves(self):
        lineage = toy_lineage(feedback=[1, 2, 1, 3], heldout=[1, 2, 3, 4])  # up/up, down/up, up/up
        self.assertAlmostEqual(direction_agreement([lineage]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
