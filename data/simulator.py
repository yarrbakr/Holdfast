"""Simulated lineages that mimic the HarnessDev evolution stage (architecture.md §6).

Generative story for one lineage
--------------------------------
1. A latent *true quality* q_t (0..100 scale) starts at q_0 and drifts slowly.
   Each edit H(t-1) -> H(t) moves it by a small random step. Edits with more
   ``revision_calls`` or with ``touched_verification=True`` are slightly more
   likely to be real improvements, as the paper reports.
2. The evolving model also *overfits the feedback set it keeps looking at*:
   every edit adds a non-negative bias o_t that inflates its score on the
   feedback tasks it was shown, but not on held-out tasks (adaptive-holdout
   overfitting). Special-casing seen tasks is not free: each point of such bias
   costs ``overfit_quality_cost`` points of true quality, so versions that
   chased the feedback set too hard generalise worse.
3. ``feedback_score`` is the pass rate x 100 on a finite feedback set of tasks,
   each task a fresh Bernoulli draw. With ~110 tasks this gives re-run noise
   with standard deviation ~4.75 points, matching the paper. A fixed fraction of
   feedback tasks is withheld from the evolving model (``selection_tasks``):
   they carry the same noise but none of the overfitting bias.
4. ``heldout_score`` is a cleaner readout of q_t (small noise only).

Default parameters were calibrated (averaging 100 seeds) to the paper's
statistics: feedback and held-out changes agree in direction ~53% of the time
and the max-feedback version is the held-out best in ~2.1 of 9 lineages
(paper: ~53% and 2 of 9). Call ``metrics.describe_dataset`` to see these
statistics for any dataset. The ``overfit_quality_cost`` assumption matters for
which rules win; see the README's "Honest caveats" section.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.contract import Lineage, Version

DEFAULT_SEED = 7


@dataclass(frozen=True)
class SimConfig:
    """All knobs of the simulator in one place. Defaults mirror the paper."""

    n_lineages: int = 9
    min_versions: int = 5  # versions per lineage, H0 included
    max_versions: int = 10
    n_feedback_tasks: int = 110  # Bernoulli pass/fail -> score sd ~ 4.75 at p ~ 0.5
    selection_fraction: float = 0.3  # share of feedback tasks withheld for selection
    start_quality_range: tuple[float, float] = (40.0, 60.0)
    quality_step_sd: float = 1.5  # random part of each edit's true effect
    quality_drift: float = 0.3  # mean true gain per edit (evolution helps a little)
    revision_calls_effect: float = 0.4  # true gain per extra revise-and-recheck cycle
    touched_verification_effect: float = 1.0  # true gain when the verifier was edited
    overfit_step_sd: float = 1.2  # scale of the per-edit feedback-only bias (half-normal)
    overfit_quality_cost: float = 0.5  # true quality lost per point of feedback-only bias
    heldout_noise_sd: float = 1.0  # held-out set is large, so it reads q_t cleanly


def simulate_lineages(seed: int = DEFAULT_SEED, config: SimConfig = SimConfig()) -> list[Lineage]:
    """Generate ``config.n_lineages`` lineages, fully determined by ``seed``."""
    rng = np.random.default_rng(seed)
    return [_simulate_one_lineage(f"sim-{i:02d}", rng, config) for i in range(config.n_lineages)]


def _simulate_one_lineage(lineage_id: str, rng: np.random.Generator, cfg: SimConfig) -> Lineage:
    """Simulate a single evolution run H0..Hn."""
    n_versions = int(rng.integers(cfg.min_versions, cfg.max_versions + 1))
    features = [_h0_features()] + [_draw_edit_features(rng) for _ in range(n_versions - 1)]

    overfit_bias = _overfit_bias_path(features, rng, cfg)
    true_quality = _true_quality_path(features, rng, cfg) - cfg.overfit_quality_cost * overfit_bias
    true_quality = np.clip(true_quality, 1.0, 99.0)
    selection_tasks = _choose_selection_tasks(rng, cfg)

    versions = []
    for t in range(n_versions):
        tasks = _draw_task_outcomes(true_quality[t], overfit_bias[t], selection_tasks, rng, cfg)
        heldout = true_quality[t] + rng.normal(0.0, cfg.heldout_noise_sd)
        versions.append(
            Version(
                feedback_score=float(100.0 * tasks.mean()),
                heldout_score=float(np.clip(heldout, 0.0, 100.0)),
                features=features[t],
                task_feedback=tuple(float(x) for x in tasks),
            )
        )
    return Lineage(
        lineage_id=lineage_id,
        versions=tuple(versions),
        selection_tasks=tuple(int(i) for i in selection_tasks),
    )


def _h0_features() -> dict:
    """H0 is the starting harness: no edit has been made yet."""
    return {"edit_files": 0, "edit_lines": 0, "touched_verification": False, "revision_calls": 0}


def _draw_edit_features(rng: np.random.Generator) -> dict:
    """Observable metadata for one edit H(t-1) -> H(t)."""
    edit_files = 1 + int(rng.poisson(1.5))
    lines_per_file = rng.lognormal(mean=np.log(30.0), sigma=0.8)
    return {
        "edit_files": edit_files,
        "edit_lines": max(1, int(round(edit_files * lines_per_file))),
        "touched_verification": bool(rng.random() < 0.3),
        "revision_calls": 1 + int(rng.poisson(2.0)),
    }


def _true_quality_path(features: list[dict], rng: np.random.Generator, cfg: SimConfig) -> np.ndarray:
    """Latent quality q_0..q_n: a slow, non-monotonic random walk nudged by features."""
    q = np.empty(len(features))
    q[0] = rng.uniform(*cfg.start_quality_range)
    for t in range(1, len(features)):
        f = features[t]
        feature_effect = (
            cfg.revision_calls_effect * (f["revision_calls"] - 3)  # 3 = mean revision_calls
            + cfg.touched_verification_effect * (float(f["touched_verification"]) - 0.3)
        )
        step = cfg.quality_drift + feature_effect + rng.normal(0.0, cfg.quality_step_sd)
        q[t] = q[t - 1] + step
    return q


def _overfit_bias_path(features: list[dict], rng: np.random.Generator, cfg: SimConfig) -> np.ndarray:
    """Cumulative feedback-only bias: each edit exploits the tasks it was shown a bit more."""
    steps = [0.0] + [abs(rng.normal(0.0, cfg.overfit_step_sd)) for _ in features[1:]]
    return np.cumsum(steps)


def _choose_selection_tasks(rng: np.random.Generator, cfg: SimConfig) -> np.ndarray:
    """Indices of feedback tasks withheld from the evolving model (fixed per lineage)."""
    n_selection = max(1, int(round(cfg.selection_fraction * cfg.n_feedback_tasks)))
    return np.sort(rng.choice(cfg.n_feedback_tasks, size=n_selection, replace=False))


def _draw_task_outcomes(
    quality: float,
    overfit: float,
    selection_tasks: np.ndarray,
    rng: np.random.Generator,
    cfg: SimConfig,
) -> np.ndarray:
    """One run of a version on the feedback set: a 0/1 outcome per task.

    Tasks the model was shown pass with probability (quality + overfit) / 100;
    withheld selection tasks pass with probability quality / 100.
    """
    p = np.full(cfg.n_feedback_tasks, (quality + overfit) / 100.0)
    p[selection_tasks] = quality / 100.0
    return (rng.random(cfg.n_feedback_tasks) < np.clip(p, 0.0, 1.0)).astype(float)
