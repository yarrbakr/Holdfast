"""Scoring a selection after the fact (architecture.md §7).

This is the ONLY module that reads ``heldout_score``. Selection rules never
see it; they are graded here once they have committed to a version.
"""

from __future__ import annotations

import numpy as np

from data.contract import Lineage


def heldout_scores(lineage: Lineage) -> np.ndarray:
    """Ground-truth held-out score of every version, in order."""
    return np.array([v.heldout_score for v in lineage.versions], dtype=float)


def oracle_best_index(lineage: Lineage) -> int:
    """Index of the version with the highest held-out score (earliest on ties)."""
    return int(np.argmax(heldout_scores(lineage)))


def held_out_regret(lineage: Lineage, chosen_index: int) -> float:
    """heldout[oracle_best] - heldout[chosen]. Zero for a perfect pick, else positive."""
    if not 0 <= chosen_index < len(lineage):
        raise IndexError(f"chosen_index {chosen_index} out of range for {len(lineage)} versions")
    scores = heldout_scores(lineage)
    return float(scores.max() - scores[chosen_index])


def win_rate(rule_regrets: np.ndarray, baseline_regrets: np.ndarray) -> float:
    """Fraction of lineages where the rule's regret is strictly below the baseline's."""
    rule_regrets = np.asarray(rule_regrets, dtype=float)
    baseline_regrets = np.asarray(baseline_regrets, dtype=float)
    if rule_regrets.shape != baseline_regrets.shape:
        raise ValueError("rule and baseline regrets must cover the same lineages")
    return float(np.mean(rule_regrets < baseline_regrets))


def direction_agreement(lineages: list[Lineage]) -> float:
    """Share of consecutive edits where feedback and held-out move the same way.

    The paper reports ~53%: the visible signal barely tracks the real one.
    """
    agree, total = 0, 0
    for lineage in lineages:
        feedback = np.array([v.feedback_score for v in lineage.versions])
        heldout = heldout_scores(lineage)
        d_feedback, d_heldout = np.sign(np.diff(feedback)), np.sign(np.diff(heldout))
        agree += int(np.sum(d_feedback == d_heldout))
        total += len(d_feedback)
    return agree / total if total else float("nan")


def describe_dataset(lineages: list[Lineage]) -> dict[str, float]:
    """Headline statistics used to check a dataset looks like the paper's setting."""
    max_feedback_is_best = [
        int(np.argmax([v.feedback_score for v in lin.versions])) == oracle_best_index(lin)
        for lin in lineages
    ]
    return {
        "n_lineages": len(lineages),
        "mean_versions": float(np.mean([len(lin) for lin in lineages])),
        "direction_agreement": direction_agreement(lineages),
        "max_feedback_is_heldout_best": int(np.sum(max_feedback_is_best)),
    }
