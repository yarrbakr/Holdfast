"""Holdfast entrypoint: load lineages, evaluate every rule, print a table, save figures.

    python run.py                          # 9 simulated lineages (seed 7) + average over 100 datasets
    python run.py --replicates 1           # just the single 9-lineage dataset (also used for the re-run sweep)
    python run.py --source real            # real lineages via data/loader.py
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.contract import Lineage, validate_lineages
from data.loader import DEFAULT_REAL_DATA_PATH, RealDataNotFound, load_lineages
from data.simulator import DEFAULT_SEED, SimConfig, simulate_lineages
from evaluate import evaluate_all, rerun_budget_sweep, summarize
from metrics import describe_dataset, heldout_scores
from rules.selection import BASELINE, DEFAULT_NOISE_SD, rerun_shortlist, rerun_top_k

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FLOOR_RULES = {BASELINE, "pick_last", "pick_h0"}  # the baselines/floors in architecture.md §5

# Colors (validated reference palette): identity of the two score series, and
# contender vs baseline bars. Text stays in neutral ink.
FEEDBACK_COLOR, HELDOUT_COLOR = "#2a78d6", "#eb6834"
CONTENDER_COLOR, FLOOR_COLOR = "#2a78d6", "#a3a29d"
INK, MUTED_INK, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4, fixed order


def load_data(source: str, seed: int, data_path: Path, sim_config: SimConfig = SimConfig()) -> list[Lineage]:
    """The one switch between data sources; both return the same contract."""
    lineages = simulate_lineages(seed, sim_config) if source == "sim" else load_lineages(data_path)
    validate_lineages(lineages)
    return lineages


def replicate_summary(datasets: list[list[Lineage]]) -> pd.DataFrame:
    """Average the per-rule summary over many independently simulated datasets."""
    n_replicates = len(datasets)
    tables = []
    for lineages in datasets:
        table = summarize(evaluate_all(lineages))
        baseline_regret = table.loc[table["rule"] == BASELINE, "mean_regret"].iloc[0]
        table["delta_vs_baseline"] = table["mean_regret"] - baseline_regret  # paired within a dataset
        tables.append(table)
    grouped = pd.concat(tables).groupby("rule", sort=False)
    return pd.DataFrame(
        {
            "mean_regret": grouped["mean_regret"].mean(),
            "se_of_mean_regret": grouped["mean_regret"].std(ddof=1) / np.sqrt(n_replicates),
            "delta_vs_baseline": grouped["delta_vs_baseline"].mean(),
            "se_of_delta": grouped["delta_vs_baseline"].std(ddof=1) / np.sqrt(n_replicates),
            "win_rate_vs_baseline": grouped["win_rate_vs_baseline"].mean(),
            "loss_rate_vs_baseline": grouped["loss_rate_vs_baseline"].mean(),
            "picked_heldout_best": grouped["picked_heldout_best"].mean(),
        }
    ).reset_index()


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #


def print_dataset_summary(lineages: list[Lineage], source: str) -> None:
    stats = describe_dataset(lineages)
    print(f"\nData source: {source}  |  {stats['n_lineages']} lineages, {stats['mean_versions']:.1f} versions on average")
    print(
        f"  feedback & held-out agree in direction: {stats['direction_agreement']:.0%}  (paper: ~53%)\n"
        f"  max-feedback version is held-out best:  {stats['max_feedback_is_heldout_best']} of "
        f"{stats['n_lineages']}  (paper: 2 of 9)"
    )


def format_table(table: pd.DataFrame, n_lineages: int | None = None) -> str:
    """Render a summary table as aligned text, showing n/a for unavailable rules."""
    shown = pd.DataFrame({"rule": table["rule"]})
    shown["mean regret"] = table["mean_regret"].map(_fmt(lambda x: f"{x:.2f}"))
    if "delta_vs_baseline" in table:
        shown["vs baseline (paired)"] = [
            "n/a" if pd.isna(d) else f"{d:+.2f} ± {se:.2f}"
            for d, se in zip(table["delta_vs_baseline"], table["se_of_delta"].fillna(0.0))
        ]
    shown["win rate vs baseline"] = table["win_rate_vs_baseline"].map(_fmt(lambda x: f"{x:.0%}"))
    shown["loss rate vs baseline"] = table["loss_rate_vs_baseline"].map(_fmt(lambda x: f"{x:.0%}"))
    if n_lineages is not None:
        shown["picked held-out best"] = table["picked_heldout_best"].map(
            _fmt(lambda x: f"{int(x)}/{n_lineages}")
        )
    else:
        shown["picked held-out best"] = table["picked_heldout_best"].map(_fmt(lambda x: f"{x:.2f}"))
    if "tuned_knob" in table:
        shown["LOLO-tuned knob"] = table["tuned_knob"]
    return shown.to_string(index=False)


def format_sweep(sweep: pd.DataFrame) -> str:
    """Render the re-run budget sweep as aligned text."""
    shown = pd.DataFrame({
        "shortlist k": sweep["k"].map(lambda k: f"{k}" if k < 10 else "all"),
        "re-runs each": sweep["n_reruns"],
        "noise sd": sweep["n_reruns"].map(lambda n: f"{DEFAULT_NOISE_SD / np.sqrt(1 + n):.1f}"),
        "extra runs/lineage": sweep["extra_runs_per_lineage"].map(lambda x: f"{x:.1f}"),
        "mean regret": sweep["mean_regret"].map(lambda x: f"{x:.2f}"),
        "vs baseline (paired)": [
            f"{d:+.2f}" + ("" if pd.isna(se) else f" ± {se:.2f}")
            for d, se in zip(sweep["delta_vs_baseline"], sweep["se_of_delta"])
        ],
    })
    return shown.to_string(index=False)


def _fmt(render):
    """Wrap a formatter so NaN/None prints as n/a."""
    return lambda x: "n/a" if x is None or pd.isna(x) else render(x)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


RERUN_COLOR = "#1baf7a"  # categorical slot 3: the re-run average (slots 1-2 are feedback / held-out)
RERUN_RULE = "rerun_top_k"


def pick_example_lineages(lineages: list[Lineage], results: pd.DataFrame) -> list[Lineage]:
    """Two lineages where the baseline misses: one re-running fixes, one it doesn't.

    Taking the FIRST lineage of each kind (not the most dramatic) keeps the
    examples from being cherry-picked. Falls back to a single example if the
    data has only one kind, or no re-run results at all.
    """
    regret = results.pivot(index="lineage_id", columns="rule", values="regret")
    misses = [lin for lin in lineages if regret.loc[lin.lineage_id, BASELINE] > 0]
    if not misses:
        return lineages[:1]
    if RERUN_RULE not in regret or regret[RERUN_RULE].isna().any():
        return misses[:1]
    improved = [lin for lin in misses if regret.loc[lin.lineage_id, RERUN_RULE] < regret.loc[lin.lineage_id, BASELINE]]
    not_improved = [lin for lin in misses if lin not in improved]
    return (improved[:1] + not_improved[:1]) or misses[:1]


def plot_trajectory(lineages: list[Lineage], results: pd.DataFrame, path: Path) -> None:
    """Figure (a): feedback vs held-out per version, with the baseline and re-run picks."""
    examples = pick_example_lineages(lineages, results)
    fig, axes = plt.subplots(len(examples), 1, figsize=(8, 4.2 * len(examples)), squeeze=False)
    for ax, lineage in zip(axes[:, 0], examples):
        _plot_one_trajectory(ax, lineage, results)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(_rerun_tally(results), x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04 if len(examples) > 1 else 0.1, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_one_trajectory(ax, lineage: Lineage, results: pd.DataFrame) -> None:
    """One panel: the two score lines, the re-run averages of the shortlist, and the picks."""
    feedback = np.array([v.feedback_score for v in lineage.versions])
    heldout = heldout_scores(lineage)
    x = np.arange(len(lineage))
    rows = results[results["lineage_id"] == lineage.lineage_id].set_index("rule")
    baseline_pick, oracle = int(rows.loc[BASELINE, "chosen_index"]), int(rows.loc[BASELINE, "oracle_index"])

    ax.plot(x, feedback, color=FEEDBACK_COLOR, lw=2, marker="o", ms=7, label="feedback score (one run)")
    ax.plot(x, heldout, color=HELDOUT_COLOR, lw=2, marker="s", ms=7, label="held-out score (ground truth)")
    _ring(ax, baseline_pick, feedback[baseline_pick], FEEDBACK_COLOR, "max-feedback pick")
    _ring(ax, oracle, heldout[oracle], HELDOUT_COLOR, "held-out best")
    title = f"{lineage.lineage_id}: max-feedback pick misses by {heldout[oracle] - heldout[baseline_pick]:.1f}"

    has_reruns = RERUN_RULE in rows.index and not pd.isna(rows.loc[RERUN_RULE, "chosen_index"])
    if has_reruns:
        defaults = inspect.signature(rerun_top_k).parameters
        k, n = defaults["k"].default, defaults["n_reruns"].default
        shortlist = rerun_shortlist(feedback, k)
        averages = {i: np.mean([feedback[i], *lineage.versions[i].rerun_scores[:n]]) for i in shortlist}
        for i, avg in averages.items():
            ax.plot([i, i], [feedback[i], avg], color=RERUN_COLOR, lw=1.2, ls=":")
        ax.scatter(list(averages), list(averages.values()), marker="D", s=55, color=RERUN_COLOR, zorder=4,
                   label=f"average of {n + 1} runs (top {k} only)")
        rerun_pick = int(rows.loc[RERUN_RULE, "chosen_index"])
        _ring(ax, rerun_pick, averages[rerun_pick], RERUN_COLOR, "re-run pick")
        title += f"; re-run pick misses by {heldout[oracle] - heldout[rerun_pick]:.1f}"
        if oracle not in shortlist:
            rank = int((feedback > feedback[oracle]).sum()) + 1
            title += f"\n(the held-out best, H{oracle}, ranked #{rank} on feedback: never shortlisted)"

    ax.set_xticks(x, [f"H{i}" for i in x])
    ax.set_xlabel("version")
    ax.set_ylabel("score (0-100)")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    _style_axes(ax)
    ax.margins(y=0.25)


def _rerun_tally(results: pd.DataFrame) -> str:
    """One-line context: how re-running did on EVERY lineage, not just the examples."""
    regret = results.pivot(index="lineage_id", columns="rule", values="regret")
    if RERUN_RULE not in regret or regret[RERUN_RULE].isna().any():
        return "The paper's rule chases the feedback peak"
    diff = regret[RERUN_RULE] - regret[BASELINE]
    better, worse = int((diff < 0).sum()), int((diff > 0).sum())
    return (f"Re-run pick vs max-feedback pick across all {len(diff)} lineages:\n"
            f"better on {better}, worse on {worse}, same on {len(diff) - better - worse}")


def _ring(ax, x: int, y: float, color: str, text: str) -> None:
    """Ring a point and label it just above."""
    ax.scatter([x], [y], s=240, facecolors="none", edgecolors=color, linewidths=2, zorder=5)
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 13), ha="center",
                fontsize=8, color=MUTED_INK)


def plot_regret_bars(table: pd.DataFrame, path: Path, subtitle: str) -> None:
    """Figure (b): mean held-out regret per rule (unavailable rules are listed, not drawn)."""
    available = table[table["mean_regret"].notna()]
    missing = table.loc[table["mean_regret"].isna(), "rule"].tolist()
    colors = [FLOOR_COLOR if r in FLOOR_RULES else CONTENDER_COLOR for r in available["rule"]]
    errors = available["se_of_mean_regret"] if "se_of_mean_regret" in available else None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(available))[::-1]
    ax.barh(y, available["mean_regret"], color=colors, height=0.6, xerr=errors,
            error_kw={"ecolor": MUTED_INK, "lw": 1, "capsize": 3})
    label_x = available["mean_regret"] + (0 if errors is None else errors.fillna(0))
    for yi, x_pos, value in zip(y, label_x, available["mean_regret"]):
        ax.text(x_pos, yi, f"  {value:.2f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y, available["rule"])
    ax.set_xlabel("mean held-out regret (points; lower is better)")
    ax.set_title(f"Mean held-out regret per rule\n{subtitle}", loc="left", fontsize=11, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (FLOOR_COLOR, CONTENDER_COLOR)]
    ax.legend(handles, ["baseline / floor", "contender"], frameon=False, loc="lower right")
    if missing:
        fig.text(0.01, 0.01, "n/a on this data (no per-task results): " + ", ".join(missing),
                 fontsize=8, color=MUTED_INK)
    _style_axes(ax, grid_axis="x")
    ax.set_xlim(0, label_x.max() * 1.2)
    _save(fig, path)


def plot_rerun_budget(sweep: pd.DataFrame, table: pd.DataFrame, path: Path, subtitle: str) -> None:
    """Figure (c): regret of rerun_top_k as the re-run budget grows, vs reference rules."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for color, (k, group) in zip(SERIES_COLORS, sweep.groupby("k", sort=True)):
        label = "re-run all versions" if k >= 10 else f"re-run top {k}"
        ax.plot(group["n_reruns"], group["mean_regret"], color=color, lw=2, marker="o", ms=7, label=label)
    for rule, style in ((BASELINE, "--"), ("shrinkage", ":")):
        value = table.loc[table["rule"] == rule, "mean_regret"].iloc[0]
        ax.axhline(value, color=MUTED_INK, lw=1.2, ls=style)
        ax.annotate(f"{rule} ({value:.2f})", (1, value), xytext=(2, 4), textcoords="offset points",
                    fontsize=9, color=MUTED_INK)
    ns = sorted(sweep["n_reruns"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("extra runs of each shortlisted version")
    ax.set_ylabel("mean held-out regret (lower is better)")
    baseline_regret = table.loc[table["rule"] == BASELINE, "mean_regret"].iloc[0]
    ax.set_ylim(0, baseline_regret * 1.15)  # headroom so the baseline label clears the title
    ax.set_title(f"Re-running the shortlist before choosing\n{subtitle}", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, loc="lower left")
    _style_axes(ax)
    _save(fig, path)


def _style_axes(ax, grid_axis: str = "y") -> None:
    """Recessive grid and axes so the data carries the ink."""
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["sim", "real"], default="sim", help="data source (default: sim)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="simulator seed (default: %(default)s)")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_REAL_DATA_PATH,
                        help="real-data file for --source real (.csv or .json)")
    parser.add_argument("--replicates", type=int, default=100,
                        help="sim only: also average over this many simulated 9-lineage datasets "
                             "(seeds seed..seed+N-1; default %(default)s; 1 disables)")
    parser.add_argument("--overfit-cost", type=float, default=SimConfig().overfit_quality_cost,
                        help="sim only: true quality lost per point of feedback overfitting "
                             "(sensitivity analysis; default %(default)s)")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sim_config = SimConfig(overfit_quality_cost=args.overfit_cost)
    try:
        lineages = load_data(args.source, args.seed, args.data_path, sim_config)
    except RealDataNotFound as err:
        print(err, file=sys.stderr)
        return 1

    source = f"simulated (seed {args.seed})" if args.source == "sim" else f"real ({args.data_path})"
    print_dataset_summary(lineages, source)

    results = evaluate_all(lineages)
    table = summarize(results)
    print(f"\nResults on this dataset ({len(lineages)} lineages; knobs tuned leave-one-lineage-out):\n")
    print(format_table(table, n_lineages=len(lineages)))

    headline, subtitle, datasets = table, f"{source}, {len(lineages)} lineages", [lineages]
    if args.source == "sim" and args.replicates > 1:
        datasets = [simulate_lineages(args.seed + r, sim_config) for r in range(args.replicates)]
        headline = replicate_summary(datasets)
        subtitle = f"averaged over {args.replicates} simulated datasets of 9 lineages (± 1 s.e.)"
        print(f"\nAveraged over {args.replicates} simulated datasets (seeds {args.seed}..{args.seed + args.replicates - 1}):\n")
        print(format_table(headline))

    figures = ["trajectory.png", "regret.png"]
    plot_trajectory(lineages, results, args.figures_dir / "trajectory.png")
    plot_regret_bars(headline, args.figures_dir / "regret.png", subtitle)

    sweep = rerun_budget_sweep(datasets)
    if sweep.empty:
        print("\nRe-run budget sweep: n/a (this data has no re-runs of each version).")
    else:
        where = f"{len(datasets)} datasets of {len(lineages)} lineages" if len(datasets) > 1 else source
        print(f"\nRe-run budget for rerun_top_k ({where}; each version's original run is averaged in):\n")
        print(format_sweep(sweep))
        plot_rerun_budget(sweep, headline, args.figures_dir / "rerun_budget.png", f"mean over {where}")
        figures.append("rerun_budget.png")
    print(f"\nSaved figures to {args.figures_dir}/: " + ", ".join(figures))
    if args.source == "sim":
        print("Note: simulated results are illustrative only until real lineages are loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
