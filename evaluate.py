"""Run every selection rule over every lineage and grade it (architecture.md §7).

Rules with a tunable knob are evaluated leave-one-lineage-out (LOLO): for each
lineage, the knob is chosen to minimise mean held-out regret on the OTHER
lineages, then applied once to the left-out lineage. A rule is therefore never
graded on a lineage that was used to tune it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.contract import Lineage
from metrics import held_out_regret, oracle_best_index, win_rate
from rules.selection import BASELINE, RULES, RuleSpec, RuleUnavailable


def visible_inputs(lineage: Lineage, spec: RuleSpec) -> dict:
    """Everything a rule is allowed to see. ``heldout_score`` is never copied in."""
    inputs = {
        "feedback_scores": np.array([v.feedback_score for v in lineage.versions], dtype=float),
        "features": tuple(dict(v.features) for v in lineage.versions),  # copies: rules can't mutate data
    }
    if spec.needs_task_data:
        has_tasks = lineage.selection_tasks is not None
        inputs["task_feedback"] = (
            np.array([v.task_feedback for v in lineage.versions], dtype=float) if has_tasks else None
        )
        inputs["selection_tasks"] = np.array(lineage.selection_tasks, dtype=int) if has_tasks else None
    if spec.needs_reruns:
        reruns = tuple(v.rerun_scores for v in lineage.versions)
        inputs["rerun_scores"] = None if all(r is None for r in reruns) else reruns
    return inputs


def choose(spec: RuleSpec, lineage: Lineage, knob_value: float | None = None, **fixed_options) -> int:
    """Apply one rule to one lineage.

    ``knob_value=None`` uses the rule's default knob. ``fixed_options`` pins other
    keyword arguments (e.g. the re-run budget in ``rerun_budget_sweep``).
    """
    options = dict(fixed_options)
    if knob_value is not None:
        options[spec.knob] = knob_value
    chosen = spec.func(**visible_inputs(lineage, spec), **options)
    if not 0 <= chosen < len(lineage):
        raise ValueError(f"rule {spec.name} returned invalid index {chosen}")
    return int(chosen)


def mean_regret(spec: RuleSpec, lineages: list[Lineage], knob_value: float | None) -> float:
    """Mean held-out regret of a rule (at a fixed knob value) over some lineages."""
    return float(np.mean([held_out_regret(lin, choose(spec, lin, knob_value)) for lin in lineages]))


def tune_knob(spec: RuleSpec, train: list[Lineage]) -> float | None:
    """Knob value with the lowest mean regret on ``train`` (first in grid on ties)."""
    if spec.knob is None or not train:
        return None
    regrets = [mean_regret(spec, train, value) for value in spec.grid]
    return spec.grid[int(np.argmin(regrets))]


def evaluate_rule(spec: RuleSpec, lineages: list[Lineage]) -> list[dict]:
    """One result row per lineage for a single rule (LOLO-tuned if it has a knob)."""
    rows = []
    for i, lineage in enumerate(lineages):
        train = lineages[:i] + lineages[i + 1 :]  # every lineage except the test one
        knob_value = tune_knob(spec, train)
        try:
            chosen = choose(spec, lineage, knob_value)
            regret = held_out_regret(lineage, chosen)
        except RuleUnavailable:
            chosen, regret = None, float("nan")
        rows.append(
            {
                "rule": spec.name,
                "lineage_id": lineage.lineage_id,
                "chosen_index": chosen,
                "oracle_index": oracle_best_index(lineage),
                "regret": regret,
                "knob": spec.knob,
                "knob_value": knob_value,
            }
        )
    return rows


def evaluate_all(lineages: list[Lineage], rules: dict[str, RuleSpec] = RULES) -> pd.DataFrame:
    """Per-(rule, lineage) results for every registered rule."""
    rows = [row for spec in rules.values() for row in evaluate_rule(spec, lineages)]
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per-rule table: mean regret, win/loss rate vs baseline, and oracle hits.

    Unavailable rules (no data for them) get NaN in every numeric column.
    """
    baseline = results[results["rule"] == BASELINE].set_index("lineage_id")["regret"]
    summary = []
    for rule, group in results.groupby("rule", sort=False):
        group = group.set_index("lineage_id")
        available = group["regret"].notna().all()
        regrets = group["regret"].to_numpy(dtype=float)
        summary.append(
            {
                "rule": rule,
                "mean_regret": float(np.mean(regrets)) if available else float("nan"),
                "win_rate_vs_baseline": (
                    win_rate(regrets, baseline.loc[group.index].to_numpy(dtype=float))
                    if available
                    else float("nan")
                ),
                # Wins + losses < 100% because a rule often ties the baseline (same pick).
                "loss_rate_vs_baseline": (
                    win_rate(baseline.loc[group.index].to_numpy(dtype=float), regrets)
                    if available
                    else float("nan")
                ),
                "picked_heldout_best": (
                    int((group["chosen_index"] == group["oracle_index"]).sum()) if available else None
                ),
                "n_lineages": len(group),
                "tuned_knob": _describe_knob(group),
            }
        )
    return pd.DataFrame(summary)


def _describe_knob(group: pd.DataFrame) -> str:
    """Human-readable summary of the LOLO-chosen knob values for one rule."""
    knob = group["knob"].iloc[0]
    if knob is None or pd.isna(knob):
        return "-"
    values = group["knob_value"].dropna()
    if values.empty:
        return f"{knob}=default"
    return f"{knob}: " + "/".join(f"{v:g}" for v in sorted(values.unique()))


def rerun_budget_sweep(
    datasets: list[list[Lineage]],
    ks: tuple[int, ...] = (2, 3, 5, 10),
    reruns: tuple[int, ...] = (1, 2, 4, 8),
) -> pd.DataFrame:
    """Regret of ``rerun_top_k`` at each re-run budget, paired against the baseline.

    For every (k, n_reruns) pair, returns the mean regret over all datasets, the
    mean paired difference from ``baseline_max_feedback`` with its standard error,
    and the average number of extra feedback runs per lineage. Returns an empty
    frame when the data has no re-runs.
    """
    spec, baseline = RULES["rerun_top_k"], RULES[BASELINE]
    baseline_regret = np.array([mean_regret(baseline, lineages, None) for lineages in datasets])
    rows = []
    for k in ks:
        for n in reruns:
            try:
                regret = np.array([
                    np.mean([held_out_regret(lin, choose(spec, lin, k=k, n_reruns=n)) for lin in lineages])
                    for lineages in datasets
                ])
            except RuleUnavailable:
                return pd.DataFrame()
            delta = regret - baseline_regret
            rows.append({
                "k": k,
                "n_reruns": n,
                "extra_runs_per_lineage": float(np.mean([min(k, len(lin)) * n for d in datasets for lin in d])),
                "mean_regret": float(regret.mean()),
                "delta_vs_baseline": float(delta.mean()),
                "se_of_delta": float(delta.std(ddof=1) / np.sqrt(len(datasets))) if len(datasets) > 1 else float("nan"),
            })
    return pd.DataFrame(rows)
