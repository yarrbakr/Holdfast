"""Load REAL lineages from disk into the Holdfast data contract (architecture.md §10).

No real lineages have been released yet, so by default this raises
``RealDataNotFound`` with instructions. When the data arrives, put it at
``data/real/lineages.csv`` (or .json) in the schema below and run
``python run.py --source real``. Nothing outside this file needs to change.

CSV schema -- ``lineages.csv``, one row per version
---------------------------------------------------
======================  =======  =============================================
column                  type     meaning
======================  =======  =============================================
lineage_id              str      evolution run the version belongs to
version_index           int      0 for H0, 1 for H1, ... (contiguous per lineage)
feedback_score          float    score on the visible feedback set
heldout_score           float    score on the held-out set (evaluation only)
edit_files              int      files changed vs the previous version
edit_lines              int      net lines changed vs the previous version
touched_verification    bool     edit modified verifier/checking logic (true/false/1/0)
revision_calls          int      revise-and-recheck cycles the model made
======================  =======  =============================================

Optional per-task file -- ``lineages_tasks.csv`` next to it (enables ``validation_split``)
---------------------------------------------------------------------------------------
One row per (version, feedback task): ``lineage_id, version_index, task_id,
score, is_selection_task``. ``score`` is that task's feedback result (0..1);
``is_selection_task`` is true for tasks that were withheld from the evolving
model and kept only for final selection. Without any withheld tasks,
``validation_split`` reports n/a.

JSON schema -- ``lineages.json``
--------------------------------
Exactly what ``save_lineages_json`` writes::

    {"lineages": [{"lineage_id": "run-1",
                   "selection_tasks": [3, 17] | null,
                   "versions": [{"feedback_score": 51.2, "heldout_score": 49.0,
                                 "features": {"edit_files": 0, "edit_lines": 0,
                                              "touched_verification": false,
                                              "revision_calls": 0},
                                 "task_feedback": [1, 0, ...] | null}, ...]}]}
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.contract import FEATURE_KEYS, Lineage, Version, validate_lineages

DEFAULT_REAL_DATA_PATH = Path(__file__).resolve().parent / "real" / "lineages.csv"

VERSION_COLUMNS = ["lineage_id", "version_index", "feedback_score", "heldout_score", *FEATURE_KEYS]
TASK_COLUMNS = ["lineage_id", "version_index", "task_id", "score", "is_selection_task"]

NO_DATA_MESSAGE = """\
No real lineage data found at: {path}

Real HarnessDev lineages have not been released yet, so Holdfast runs on
simulated data by default (python run.py --source sim).

To use real data, write one row per version to data/real/lineages.csv with
columns:
  {columns}
(optionally add data/real/lineages_tasks.csv for per-task feedback), or supply a
.json file; see the docstring of data/loader.py for the full schema. Then run:
  python run.py --source real [--data-path <file>]"""


class RealDataNotFound(FileNotFoundError):
    """Raised when the real-data file does not exist yet."""


def load_lineages(path: str | Path = DEFAULT_REAL_DATA_PATH) -> list[Lineage]:
    """Read real lineages from a .csv or .json file and validate them against the contract."""
    path = Path(path)
    if not path.exists():
        raise RealDataNotFound(NO_DATA_MESSAGE.format(path=path, columns=", ".join(VERSION_COLUMNS)))
    if path.suffix.lower() == ".json":
        lineages = _load_json(path)
    elif path.suffix.lower() == ".csv":
        lineages = _load_csv(path)
    else:
        raise ValueError(f"unsupported file type {path.suffix!r}; use .csv or .json")
    validate_lineages(lineages)
    return lineages


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def _load_csv(path: Path) -> list[Lineage]:
    """Parse the per-version CSV (plus the optional per-task CSV beside it)."""
    versions = pd.read_csv(path, float_precision="round_trip")
    _require_columns(versions, VERSION_COLUMNS, path)
    tasks_path = path.with_name(f"{path.stem}_tasks.csv")
    tasks = pd.read_csv(tasks_path, float_precision="round_trip") if tasks_path.exists() else None
    if tasks is not None:
        _require_columns(tasks, TASK_COLUMNS, tasks_path)

    lineages = []
    for lineage_id, rows in versions.groupby("lineage_id", sort=False):
        rows = rows.sort_values("version_index")
        _require_contiguous(rows["version_index"].tolist(), lineage_id)
        task_matrix, selection = _task_data_for(tasks, lineage_id, len(rows))
        lineages.append(
            Lineage(
                lineage_id=str(lineage_id),
                versions=tuple(
                    _version_from_row(row, None if task_matrix is None else task_matrix[i])
                    for i, (_, row) in enumerate(rows.iterrows())
                ),
                selection_tasks=selection,
            )
        )
    return lineages


def _version_from_row(row: pd.Series, task_row: tuple[float, ...] | None) -> Version:
    """Build one Version from a CSV row."""
    return Version(
        feedback_score=float(row["feedback_score"]),
        heldout_score=float(row["heldout_score"]),
        features={
            "edit_files": int(row["edit_files"]),
            "edit_lines": int(row["edit_lines"]),
            "touched_verification": _parse_bool(row["touched_verification"]),
            "revision_calls": int(row["revision_calls"]),
        },
        task_feedback=task_row,
    )


def _task_data_for(
    tasks: pd.DataFrame | None, lineage_id: str, n_versions: int
) -> tuple[list[tuple[float, ...]] | None, tuple[int, ...] | None]:
    """Per-version task scores (tasks ordered by task_id) and withheld-task indices."""
    if tasks is None:
        return None, None
    rows = tasks[tasks["lineage_id"] == lineage_id]
    if rows.empty:
        return None, None
    matrix = rows.pivot(index="version_index", columns="task_id", values="score").sort_index()
    if list(matrix.index) != list(range(n_versions)) or matrix.isna().any().any():
        raise ValueError(f"lineage {lineage_id!r}: per-task file must cover every version and task")
    flags = rows.groupby("task_id")["is_selection_task"].agg(lambda s: {_parse_bool(x) for x in s})
    if any(len(v) != 1 for v in flags):
        raise ValueError(f"lineage {lineage_id!r}: is_selection_task must be constant per task")
    is_selection = [next(iter(flags[task])) for task in matrix.columns]
    selection = tuple(i for i, flag in enumerate(is_selection) if flag) or None
    return [tuple(float(x) for x in r) for r in matrix.to_numpy()], selection


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def _load_json(path: Path) -> list[Lineage]:
    """Parse the JSON layout documented at the top of this module."""
    payload = json.loads(path.read_text())
    return [_lineage_from_dict(item) for item in payload["lineages"]]


def _lineage_from_dict(item: dict) -> Lineage:
    """Build one Lineage from its JSON dict."""
    versions = tuple(
        Version(
            feedback_score=float(v["feedback_score"]),
            heldout_score=float(v["heldout_score"]),
            features=dict(v["features"]),
            task_feedback=None if v.get("task_feedback") is None else tuple(map(float, v["task_feedback"])),
        )
        for v in item["versions"]
    )
    selection = item.get("selection_tasks")
    return Lineage(
        lineage_id=str(item["lineage_id"]),
        versions=versions,
        selection_tasks=None if selection is None else tuple(int(i) for i in selection),
    )


def save_lineages_json(lineages: list[Lineage], path: str | Path) -> None:
    """Write lineages in the JSON schema above (handy as a template for real data)."""
    payload = {
        "lineages": [
            {
                "lineage_id": lin.lineage_id,
                "selection_tasks": None if lin.selection_tasks is None else list(lin.selection_tasks),
                "versions": [
                    {
                        "feedback_score": v.feedback_score,
                        "heldout_score": v.heldout_score,
                        "features": dict(v.features),
                        "task_feedback": None if v.task_feedback is None else list(v.task_feedback),
                    }
                    for v in lin.versions
                ],
            }
            for lin in lineages
        ]
    }
    Path(path).write_text(json.dumps(payload, indent=1))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def _require_contiguous(indices: list[int], lineage_id: str) -> None:
    if indices != list(range(len(indices))):
        raise ValueError(f"lineage {lineage_id!r}: version_index must be 0..n with no gaps, got {indices}")


def _parse_bool(value: object) -> bool:
    """Accept true/false, yes/no, 1/0 (any case) as booleans."""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "1.0"}:
        return True
    if text in {"false", "0", "no", "0.0"}:
        return False
    raise ValueError(f"cannot parse {value!r} as a boolean")
