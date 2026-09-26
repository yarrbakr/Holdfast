"""No selection rule may read heldout_score (architecture.md §3, §8)."""

import dataclasses
import inspect
import unittest

from data.simulator import simulate_lineages
from evaluate import choose, evaluate_all, evaluate_rule, visible_inputs
from rules.selection import RULES


class HeldoutAccessed(AssertionError):
    pass


class GuardedVersion:
    """Looks like a Version, but touching heldout_score fails the test."""

    def __init__(self, version):
        self.feedback_score = version.feedback_score
        self.features = version.features
        self.task_feedback = version.task_feedback
        self.rerun_scores = version.rerun_scores

    @property
    def heldout_score(self):
        raise HeldoutAccessed("a selection rule tried to read heldout_score")


def guarded(lineage):
    """Same lineage with every version's heldout_score behind a tripwire."""
    return dataclasses.replace(lineage, versions=tuple(GuardedVersion(v) for v in lineage.versions))


def with_heldout(lineage, new_scores):
    """Copy of ``lineage`` with its held-out scores replaced."""
    versions = tuple(dataclasses.replace(v, heldout_score=float(s)) for v, s in zip(lineage.versions, new_scores))
    return dataclasses.replace(lineage, versions=versions)


class NoPeekingTest(unittest.TestCase):
    def setUp(self):
        self.lineages = simulate_lineages(seed=123)

    def test_no_rule_signature_mentions_heldout(self):
        for name, spec in RULES.items():
            params = inspect.signature(spec.func).parameters
            self.assertEqual(list(params)[:2], ["feedback_scores", "features"], name)
            self.assertFalse(any("heldout" in p.lower() for p in params), name)

    def test_visible_inputs_never_include_heldout(self):
        for spec in RULES.values():
            inputs = visible_inputs(self.lineages[0], spec)
            self.assertFalse(any("heldout" in key for key in inputs))

    def test_rules_run_with_heldout_behind_a_tripwire(self):
        # Every rule, at every knob value, must choose without touching heldout_score.
        for lineage in map(guarded, self.lineages):
            for spec in RULES.values():
                for knob_value in (None, *spec.grid):
                    chosen = choose(spec, lineage, knob_value)
                    self.assertTrue(0 <= chosen < len(lineage.versions))

    def test_test_lineage_heldout_cannot_change_its_choice(self):
        # Under leave-one-lineage-out, rewriting the TEST lineage's held-out scores
        # (even reversing them) must leave that lineage's chosen version unchanged.
        original = evaluate_all(self.lineages).set_index(["rule", "lineage_id"])["chosen_index"]
        for i, lineage in enumerate(self.lineages):
            reversed_scores = [v.heldout_score for v in lineage.versions][::-1]
            tampered = list(self.lineages)
            tampered[i] = with_heldout(lineage, [100.0 - s for s in reversed_scores])
            for spec in RULES.values():
                row = evaluate_rule(spec, tampered)[i]
                self.assertEqual(row["chosen_index"], original[(spec.name, lineage.lineage_id)], spec.name)


if __name__ == "__main__":
    unittest.main()
