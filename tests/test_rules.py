"""Behaviour of individual selection rules on hand-built inputs (architecture.md §5)."""

import unittest

import numpy as np

from evaluate import tune_knob
from data.simulator import simulate_lineages
from rules.selection import (
    RULES,
    RuleUnavailable,
    baseline_max_feedback,
    one_standard_error,
    pick_h0,
    pick_last,
    rerun_top_k,
    shrinkage,
    thresholdout,
    validation_split,
)

NO_FEATURES = tuple({} for _ in range(10))  # these rules ignore features


class BaselineTest(unittest.TestCase):
    def test_baselines(self):
        scores = np.array([50.0, 58.0, 52.0, 58.0])
        self.assertEqual(baseline_max_feedback(scores, NO_FEATURES), 1)  # earliest on ties
        self.assertEqual(pick_last(scores, NO_FEATURES), 3)
        self.assertEqual(pick_h0(scores, NO_FEATURES), 0)


class OneStandardErrorTest(unittest.TestCase):
    def test_picks_earliest_version_within_one_se_of_best(self):
        scores = np.array([50.0, 56.0, 53.0, 60.0])
        self.assertEqual(one_standard_error(scores, NO_FEATURES, noise_sd=4.75), 1)  # 56 >= 60 - 4.75

    def test_clear_winner_is_kept(self):
        scores = np.array([40.0, 45.0, 60.0])
        self.assertEqual(one_standard_error(scores, NO_FEATURES, noise_sd=4.75), 2)


class ThresholdoutTest(unittest.TestCase):
    def test_ignores_gains_inside_the_noise_band(self):
        scores = np.array([50.0, 53.0, 54.0, 51.0])
        self.assertEqual(thresholdout(scores, NO_FEATURES, threshold=1.0, noise_sd=4.75), 0)

    def test_switches_on_a_real_gain(self):
        scores = np.array([50.0, 53.0, 60.0, 61.0])
        self.assertEqual(thresholdout(scores, NO_FEATURES, threshold=1.0, noise_sd=4.75), 2)

    def test_zero_threshold_follows_every_new_high(self):
        scores = np.array([50.0, 53.0, 52.0, 58.0, 57.0])
        self.assertEqual(thresholdout(scores, NO_FEATURES, threshold=0.0), baseline_max_feedback(scores, NO_FEATURES))


class ValidationSplitTest(unittest.TestCase):
    def test_uses_only_withheld_tasks(self):
        # Task 0-1 were shown to the evolver (version 1 aces them); tasks 2-3 were withheld.
        tasks = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float)
        scores = 100 * tasks.mean(axis=1)
        chosen = validation_split(scores, NO_FEATURES, task_feedback=tasks, selection_tasks=np.array([2, 3]))
        self.assertEqual(chosen, 0)

    def test_unavailable_without_task_data(self):
        with self.assertRaises(RuleUnavailable):
            validation_split(np.array([1.0, 2.0]), NO_FEATURES)


class ShrinkageTest(unittest.TestCase):
    def test_zero_correlation_matches_argmax_when_signal_is_clear(self):
        scores = np.array([40.0, 70.0, 45.0, 60.0])  # spread far above the noise
        self.assertEqual(shrinkage(scores, NO_FEATURES, correlation=0.0), 1)

    def test_pooling_neighbours_flattens_an_isolated_spike(self):
        # H2 is a lone spike; H4-H6 are consistently good.
        scores = np.array([50.0, 50.0, 62.0, 50.0, 58.0, 59.0, 58.0])
        self.assertEqual(baseline_max_feedback(scores, NO_FEATURES), 2)
        self.assertIn(shrinkage(scores, NO_FEATURES, correlation=0.9), (4, 5, 6))


class ExplodingReruns(tuple):
    """Re-runs of a version the rule did NOT shortlist: reading them is a budget violation."""

    def __getitem__(self, item):
        raise AssertionError("rerun_top_k read re-runs of a version outside its shortlist")

    def __len__(self):
        raise AssertionError("rerun_top_k read re-runs of a version outside its shortlist")


class RerunTopKTest(unittest.TestCase):
    def test_re_runs_overturn_a_lucky_spike(self):
        scores = np.array([50.0, 62.0, 57.0])  # H1 looks best, but its re-runs are mediocre
        reruns = ((50.0,) * 4, (51.0, 52.0, 50.0, 53.0), (58.0, 57.0, 59.0, 58.0))
        self.assertEqual(baseline_max_feedback(scores, NO_FEATURES), 1)
        self.assertEqual(rerun_top_k(scores, NO_FEATURES, rerun_scores=reruns, k=2, n_reruns=4), 2)

    def test_only_reads_the_shortlist_and_the_budget(self):
        scores = np.array([50.0, 62.0, 57.0, 40.0])
        reruns = (ExplodingReruns(), (60.0, 61.0, 99.0), (57.0, 58.0, 0.0), ExplodingReruns())
        # k=2 shortlists H1 and H2; n_reruns=2 must ignore each version's third re-run.
        self.assertEqual(rerun_top_k(scores, NO_FEATURES, rerun_scores=reruns, k=2, n_reruns=2), 1)

    def test_shortlist_of_one_is_the_baseline(self):
        scores = np.array([50.0, 62.0, 57.0])
        reruns = ((70.0,), (40.0,), (70.0,))
        self.assertEqual(rerun_top_k(scores, NO_FEATURES, rerun_scores=reruns, k=1, n_reruns=1), 1)

    def test_unavailable_without_enough_re_runs(self):
        scores = np.array([50.0, 62.0])
        with self.assertRaises(RuleUnavailable):
            rerun_top_k(scores, NO_FEATURES)
        with self.assertRaises(RuleUnavailable):
            rerun_top_k(scores, NO_FEATURES, rerun_scores=((1.0,), (1.0,)), k=2, n_reruns=4)


class RegistryTest(unittest.TestCase):
    def test_all_architecture_rules_are_registered(self):
        expected = {
            "baseline_max_feedback", "pick_last", "pick_h0", "one_standard_error",
            "thresholdout", "validation_split", "shrinkage", "rerun_top_k",
        }
        self.assertEqual(set(RULES), expected)

    def test_tuned_knob_comes_from_the_grid(self):
        lineages = simulate_lineages(seed=2)
        for spec in RULES.values():
            value = tune_knob(spec, lineages)
            if spec.knob is None:
                self.assertIsNone(value)
            else:
                self.assertIn(value, spec.grid)


if __name__ == "__main__":
    unittest.main()
