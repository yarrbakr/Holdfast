"""The Holdfast data contract (architecture.md §3).

Every data source -- the simulator today, the real-data loader later -- returns
a ``list[Lineage]`` built from the two classes below. Selection rules, metrics
and evaluation only ever depend on this module, which is what makes swapping
simulated data for real data a one-line change.

A *lineage* is one evolution run: an ordered tuple of versions H0..Hn.
A *version* carries:

* ``feedback_score``  -- visible during development; selection rules may use it.
* ``heldout_score``   -- ground truth; used ONLY by ``metrics.py`` for grading.
* ``features``        -- observable edit metadata; selection rules may use it.

Optional fields enable rules that need more than one score per version. Each
is ``None`` whenever a data source does not have that data:

* ``Version.task_feedback``   -- per-task feedback outcomes (0..1 each) whose
  mean x 100 is the version's ``feedback_score``.
* ``Lineage.selection_tasks`` -- indices into ``task_feedback`` of the tasks
  that were withheld from the evolving model and kept purely for selection.
  Together with ``task_feedback`` this enables ``validation_split``.
* ``Version.rerun_scores``    -- extra, independent re-runs of the frozen
  version on the same feedback set (each one scored like ``feedback_score``).
  Enables ``rerun_top_k``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# The exact feature keys and their types (architecture.md §3).
FEATURE_TYPES: dict[str, type] = {
    "edit_files": int,  # files changed vs the previous version
    "edit_lines": int,  # net lines changed
    "touched_verification": bool,  # did the edit modify verifier/checking logic
    "revision_calls": int,  # revise-and-recheck cycles the model made
}
FEATURE_KEYS: tuple[str, ...] = tuple(FEATURE_TYPES)


@dataclass(frozen=True)
class Version:
    """One harness version Hi inside a lineage."""

    feedback_score: float
    heldout_score: float
    features: dict[str, Any]
    task_feedback: tuple[float, ...] | None = None
    rerun_scores: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Lineage:
    """One evolution run: versions H0..Hn in order."""

    lineage_id: str
    versions: tuple[Version, ...]
    selection_tasks: tuple[int, ...] | None = None

    def __len__(self) -> int:
        return len(self.versions)


def validate_features(features: dict[str, Any]) -> None:
    """Raise ValueError unless ``features`` has exactly the contract keys and types."""
    if set(features) != set(FEATURE_KEYS):
        raise ValueError(f"features must have keys {sorted(FEATURE_KEYS)}, got {sorted(features)}")
    for key, expected in FEATURE_TYPES.items():
        value = features[key]
        # bool is a subclass of int in Python, so check the int fields strictly.
        if expected is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"feature {key!r} must be int, got {type(value).__name__}")
        if expected is bool and not isinstance(value, bool):
            raise ValueError(f"feature {key!r} must be bool, got {type(value).__name__}")


def validate_lineage(lineage: Lineage) -> None:
    """Raise ValueError if ``lineage`` breaks the data contract in any way."""
    if not isinstance(lineage, Lineage):
        raise ValueError(f"expected Lineage, got {type(lineage).__name__}")
    if len(lineage.versions) < 1:
        raise ValueError(f"lineage {lineage.lineage_id!r} has no versions")

    for i, version in enumerate(lineage.versions):
        where = f"lineage {lineage.lineage_id!r} version H{i}"
        if not isinstance(version, Version):
            raise ValueError(f"{where}: expected Version, got {type(version).__name__}")
        for name in ("feedback_score", "heldout_score"):
            value = getattr(version, name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{where}: {name} must be a finite float, got {value!r}")
        try:
            validate_features(version.features)
        except ValueError as err:
            raise ValueError(f"{where}: {err}") from None
        if version.rerun_scores is not None and not all(
            isinstance(x, float) and math.isfinite(x) for x in version.rerun_scores
        ):
            raise ValueError(f"{where}: rerun_scores must be finite floats")

    _validate_task_data(lineage)


def _validate_task_data(lineage: Lineage) -> None:
    """Check the optional per-task fields are either all absent or consistent."""
    task_rows = [v.task_feedback for v in lineage.versions]
    if lineage.selection_tasks is None:
        return  # task_feedback alone is allowed (it just doesn't enable validation_split)
    if any(row is None for row in task_rows):
        raise ValueError(f"lineage {lineage.lineage_id!r}: selection_tasks set but task_feedback missing")
    n_tasks = {len(row) for row in task_rows}
    if len(n_tasks) != 1:
        raise ValueError(f"lineage {lineage.lineage_id!r}: versions have different task counts")
    (n,) = n_tasks
    if not lineage.selection_tasks or not all(0 <= t < n for t in lineage.selection_tasks):
        raise ValueError(f"lineage {lineage.lineage_id!r}: selection_tasks out of range")


def validate_lineages(lineages: list[Lineage]) -> None:
    """Validate a whole dataset: a non-empty list of contract-conforming lineages."""
    if not isinstance(lineages, list) or not lineages:
        raise ValueError("a data source must return a non-empty list of Lineage objects")
    for lineage in lineages:
        validate_lineage(lineage)
    ids = [lin.lineage_id for lin in lineages]
    if len(set(ids)) != len(ids):
        raise ValueError("lineage_id values must be unique")
