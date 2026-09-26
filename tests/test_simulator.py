"""Simulator output honours the data contract and matches the paper (architecture.md §3, §6, §8)."""

import dataclasses
import unittest

import numpy as np

from data.contract import FEATURE_KEYS, Lineage, Version, validate_lineage, validate_lineages
from data.simulator import SimConfig, simulate_lineages
from evaluate import evaluate_all
from metrics import describe_dataset


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.lineages = simulate_lineages(seed=11)

    def test_passes_contract_validator(self):
        validate_lineages(self.lineages)  # raises on any violation

    def test_shape_and_types(self):
        self.assertIsInstance(self.lineages, list)
        self.assertEqual(len(self.lineages), 9)
        for lineage in self.lineages:
            self.assertIsInstance(lineage, Lineage)
            self.assertTrue(5 <= len(lineage.versions) <= 10)
            for version in lineage.versions:
                self.assertIsInstance(version, Version)
                self.assertIsInstance(version.feedback_score, float)
                self.assertIsInstance(version.heldout_score, float)
                self.assertEqual(tuple(version.features), FEATURE_KEYS)
                self.assertIs(type(version.features["touched_verification"]), bool)
                for key in ("edit_files", "edit_lines", "revision_calls"):
                    self.assertIs(type(version.features[key]), int)

    def test_h0_has_no_edit(self):
        for lineage in self.lineages:
            h0 = lineage.versions[0].features
            self.assertEqual((h0["edit_files"], h0["edit_lines"], h0["revision_calls"]), (0, 0, 0))

    def test_feedback_score_is_mean_of_task_feedback(self):
        for lineage in self.lineages:
            for version in lineage.versions:
                self.assertAlmostEqual(version.feedback_score, 100 * np.mean(version.task_feedback))

    def test_validator_rejects_broken_lineages(self):
        good = self.lineages[0]
        bad_features = dataclasses.replace(
            good.versions[0], features={**good.versions[0].features, "edit_files": 1.5}
        )
        missing_key = dataclasses.replace(good.versions[0], features={"edit_files": 0})
        bad_score = dataclasses.replace(good.versions[0], heldout_score=float("nan"))
        for broken in (bad_features, missing_key, bad_score):
            with self.assertRaises(ValueError):
                validate_lineage(dataclasses.replace(good, versions=(broken, *good.versions[1:])))


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_lineages_and_results(self):
        a, b = simulate_lineages(seed=3), simulate_lineages(seed=3)
        self.assertEqual(a, b)
        self.assertTrue(evaluate_all(a).equals(evaluate_all(b)))

    def test_different_seed_different_lineages(self):
        self.assertNotEqual(simulate_lineages(seed=3), simulate_lineages(seed=4))


class PaperCalibrationTest(unittest.TestCase):
    """Averaged over seeds, the simulator should reproduce the paper's headline statistics."""

    def setUp(self):
        self.datasets = [simulate_lineages(seed=s) for s in range(40)]

    def test_direction_agreement_near_half(self):
        agreement = np.mean([describe_dataset(d)["direction_agreement"] for d in self.datasets])
        self.assertTrue(0.45 <= agreement <= 0.60, agreement)

    def test_feedback_noise_sd_near_4_75(self):
        cfg = SimConfig()
        # Re-run noise of one version's score is binomial: 100 * sqrt(p (1 - p) / n_tasks).
        noise_sds = [
            100 * np.sqrt(p * (1 - p) / cfg.n_feedback_tasks)
            for d in self.datasets for lin in d for p in [lin.versions[0].feedback_score / 100]
        ]
        self.assertTrue(4.3 <= np.mean(noise_sds) <= 5.0, np.mean(noise_sds))

    def test_max_feedback_rarely_the_heldout_best(self):
        hits = np.mean([describe_dataset(d)["max_feedback_is_heldout_best"] for d in self.datasets])
        self.assertTrue(1.0 <= hits <= 3.5, hits)  # paper: 2 of 9

    def test_trajectories_are_non_monotonic(self):
        for lineage in self.datasets[0]:
            steps = np.diff([v.feedback_score for v in lineage.versions])
            self.assertTrue((steps > 0).any() and (steps < 0).any(), lineage.lineage_id)


if __name__ == "__main__":
    unittest.main()
