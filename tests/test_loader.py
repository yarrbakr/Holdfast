"""The real-data loader returns the same contract as the simulator (architecture.md §10)."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.contract import FEATURE_KEYS
from data.loader import RealDataNotFound, load_lineages, save_lineages_json
from data.simulator import simulate_lineages
from evaluate import evaluate_all


def write_csvs(lineages, directory: Path) -> Path:
    """Write lineages in the documented CSV schema (versions + per-task file)."""
    version_rows, task_rows = [], []
    for lin in lineages:
        selection = set(lin.selection_tasks)
        for i, v in enumerate(lin.versions):
            version_rows.append(
                {"lineage_id": lin.lineage_id, "version_index": i,
                 "feedback_score": v.feedback_score, "heldout_score": v.heldout_score,
                 **{k: v.features[k] for k in FEATURE_KEYS}}
            )
            for task_id, score in enumerate(v.task_feedback):
                task_rows.append(
                    {"lineage_id": lin.lineage_id, "version_index": i, "task_id": task_id,
                     "score": score, "is_selection_task": task_id in selection}
                )
    path = directory / "lineages.csv"
    pd.DataFrame(version_rows).to_csv(path, index=False)
    pd.DataFrame(task_rows).to_csv(directory / "lineages_tasks.csv", index=False)
    return path


class LoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.lineages = simulate_lineages(seed=9)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_gives_clear_message(self):
        with self.assertRaises(RealDataNotFound) as ctx:
            load_lineages(self.dir / "nope.csv")
        self.assertIn("run.py --source real", str(ctx.exception))

    def test_json_round_trip_is_identical(self):
        path = self.dir / "lineages.json"
        save_lineages_json(self.lineages, path)
        self.assertEqual(load_lineages(path), self.lineages)

    def test_csv_round_trip_gives_identical_results(self):
        loaded = load_lineages(write_csvs(self.lineages, self.dir))
        self.assertEqual(loaded, self.lineages)
        self.assertTrue(evaluate_all(loaded).equals(evaluate_all(self.lineages)))

    def test_csv_without_task_file_disables_validation_split_only(self):
        path = write_csvs(self.lineages, self.dir)
        (self.dir / "lineages_tasks.csv").unlink()
        results = evaluate_all(load_lineages(path))
        by_rule = results.groupby("rule")["regret"]
        self.assertTrue(by_rule.apply(lambda s: s.isna().all())["validation_split"])
        self.assertFalse(results[results["rule"] != "validation_split"]["regret"].isna().any())

    def test_bad_version_index_rejected(self):
        path = write_csvs(self.lineages, self.dir)
        frame = pd.read_csv(path)
        frame.loc[0, "version_index"] = 99
        frame.to_csv(path, index=False)
        (self.dir / "lineages_tasks.csv").unlink()
        with self.assertRaises(ValueError):
            load_lineages(path)


if __name__ == "__main__":
    unittest.main()
