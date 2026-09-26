"""Selection rules: which version of a lineage should be declared final?

Every rule has the signature (architecture.md §5)::

    rule(feedback_scores, features, **keyword_only_options) -> chosen_index

* ``feedback_scores`` -- 1-D numpy array, one visible score per version H0..Hn.
* ``features``        -- tuple of per-version feature dicts (see data/contract.py).

Rules never receive a Lineage object, so they have no way to reach
``heldout_score``; ``evaluate.visible_inputs`` builds their inputs. Ties are
always broken toward the EARLIEST version (the less-edited, less-overfit one).

Adding a rule = writing one function decorated with ``@register_rule``. If it
has a tunable knob, name the keyword argument and its search grid in the
decorator; evaluate.py will tune it with leave-one-lineage-out automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Re-running the same frozen harness swings its feedback score by ~4.75 points
# (the paper's measurement). Rules use it as the default noise level of one score.
DEFAULT_NOISE_SD = 4.75

BASELINE = "baseline_max_feedback"


class RuleUnavailable(Exception):
    """Raised by a rule whose required inputs are absent from this dataset."""


@dataclass(frozen=True)
class RuleSpec:
    """Registry entry describing how evaluate.py should run a rule."""

    name: str
    func: Callable[..., int]
    knob: str | None = None  # name of the tunable keyword argument, if any
    grid: tuple[float, ...] = field(default_factory=tuple)  # candidate knob values
    needs_task_data: bool = False  # also receives task_feedback / selection_tasks
    needs_reruns: bool = False  # also receives rerun_scores


RULES: dict[str, RuleSpec] = {}


def register_rule(
    knob: str | None = None,
    grid: tuple[float, ...] = (),
    needs_task_data: bool = False,
    needs_reruns: bool = False,
) -> Callable[[Callable[..., int]], Callable[..., int]]:
    """Decorator that adds a rule to ``RULES`` (registration order = table order)."""

    def decorator(func: Callable[..., int]) -> Callable[..., int]:
        if (knob is None) != (len(grid) == 0):
            raise ValueError(f"{func.__name__}: a knob needs a grid and vice versa")
        RULES[func.__name__] = RuleSpec(
            func.__name__, func, knob, tuple(grid), needs_task_data, needs_reruns
        )
        return func

    return decorator


def _first_argmax(values: np.ndarray) -> int:
    """Index of the maximum value, earliest on ties."""
    return int(np.argmax(values))


# --------------------------------------------------------------------------- #
# Baselines -- the floor to beat
# --------------------------------------------------------------------------- #


@register_rule()
def baseline_max_feedback(feedback_scores: np.ndarray, features: tuple[dict, ...]) -> int:
    """The paper's rule: declare the version with the highest feedback score."""
    return _first_argmax(np.asarray(feedback_scores, dtype=float))


@register_rule()
def pick_last(feedback_scores: np.ndarray, features: tuple[dict, ...]) -> int:
    """Always ship the final version Hn."""
    return len(feedback_scores) - 1


@register_rule()
def pick_h0(feedback_scores: np.ndarray, features: tuple[dict, ...]) -> int:
    """Always ship the starting harness H0 (i.e. evolution changes nothing)."""
    return 0


# --------------------------------------------------------------------------- #
# Contenders
# --------------------------------------------------------------------------- #


@register_rule()
def one_standard_error(
    feedback_scores: np.ndarray,
    features: tuple[dict, ...],
    *,
    noise_sd: float = DEFAULT_NOISE_SD,
) -> int:
    """Earliest version whose score is within one standard error of the best.

    Versions inside the noise band of the leader are statistically tied with it,
    so prefer the earliest (least-edited) of them.
    """
    scores = np.asarray(feedback_scores, dtype=float)
    tied_with_best = scores >= scores.max() - noise_sd
    return int(np.flatnonzero(tied_with_best)[0])


@register_rule(knob="threshold", grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0))
def thresholdout(
    feedback_scores: np.ndarray,
    features: tuple[dict, ...],
    *,
    threshold: float = 1.0,
    noise_sd: float = DEFAULT_NOISE_SD,
) -> int:
    """Walk H0..Hn keeping an incumbent; switch only on a gain above the noise threshold.

    ``threshold`` is in units of ``noise_sd``: a challenger replaces the
    incumbent only if it beats it by more than ``threshold * noise_sd`` points,
    so single lucky spikes inside the noise band are ignored. (A deterministic
    reading of Dwork et al.'s Thresholdout, without its added Laplace noise.)
    """
    scores = np.asarray(feedback_scores, dtype=float)
    incumbent = 0
    for t in range(1, len(scores)):
        if scores[t] - scores[incumbent] > threshold * noise_sd:
            incumbent = t
    return incumbent


@register_rule(needs_task_data=True)
def validation_split(
    feedback_scores: np.ndarray,
    features: tuple[dict, ...],
    *,
    task_feedback: np.ndarray | None = None,
    selection_tasks: np.ndarray | None = None,
) -> int:
    """Pick the best version on the feedback tasks that were withheld from the evolver.

    Those tasks never drove any edit, so their score is an honest (if noisier)
    signal free of adaptive overfitting. Needs per-task feedback results and the
    list of withheld tasks; raises RuleUnavailable when a dataset lacks them.
    """
    if task_feedback is None or selection_tasks is None:
        raise RuleUnavailable("validation_split needs per-task feedback with withheld selection tasks")
    per_task = np.asarray(task_feedback, dtype=float)  # shape (n_versions, n_tasks)
    selection_scores = per_task[:, np.asarray(selection_tasks, dtype=int)].mean(axis=1)
    return _first_argmax(selection_scores)


@register_rule(knob="correlation", grid=(0.0, 0.25, 0.5, 0.75, 0.9))
def shrinkage(
    feedback_scores: np.ndarray,
    features: tuple[dict, ...],
    *,
    correlation: float = 0.5,
    noise_sd: float = DEFAULT_NOISE_SD,
) -> int:
    """Empirical-Bayes shrinkage toward the lineage mean, then argmax.

    Model: score_t = quality_t + noise (sd ``noise_sd``), with a Gaussian prior
    quality ~ N(lineage_mean, tau^2 * K) where K[i, j] = correlation^|i - j|
    (neighbouring versions have similar true quality). tau^2 is estimated by
    method of moments: observed spread minus noise variance. The posterior mean
    shrinks every score toward the mean by an amount set by the noise.

    With correlation = 0 this is plain James-Stein shrinkage, which rescales all
    scores by the same factor and so never changes the argmax (except when the
    estimated signal is zero, when everything ties and H0 is kept). A positive
    correlation also pools each version with its neighbours, which is what
    flattens isolated lucky spikes. The knob is tuned leave-one-lineage-out.
    """
    shrunk = shrunk_scores(feedback_scores, correlation=correlation, noise_sd=noise_sd)
    return _first_argmax(shrunk)


def shrunk_scores(
    feedback_scores: np.ndarray,
    *,
    correlation: float = 0.5,
    noise_sd: float = DEFAULT_NOISE_SD,
) -> np.ndarray:
    """Posterior-mean quality of each version under the ``shrinkage`` model."""
    scores = np.asarray(feedback_scores, dtype=float)
    n = len(scores)
    mean = scores.mean()
    if n < 2:
        return scores.copy()
    lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    prior_corr = correlation**lags  # K; note 0**0 == 1, so correlation=0 gives the identity
    signal_var = scores.var(ddof=1) - noise_sd**2
    if signal_var <= 0.0:
        # Moment estimate says "no real spread": the posterior collapses onto the
        # mean. Keep the ordering of its tau^2 -> 0 limit, K @ (scores - mean),
        # scaled to be negligible, so the rule still prefers the best neighbourhood.
        return mean + 1e-9 * prior_corr @ (scores - mean)
    prior_cov = signal_var * prior_corr
    gain = prior_cov @ np.linalg.inv(prior_cov + noise_sd**2 * np.eye(n))
    return mean + gain @ (scores - mean)


@register_rule(needs_reruns=True)
def rerun_top_k(
    feedback_scores: np.ndarray,
    features: tuple[dict, ...],
    *,
    rerun_scores: tuple[tuple[float, ...] | None, ...] | None = None,
    k: int = 3,
    n_reruns: int = 4,
) -> int:
    """Re-run the k best-looking versions n_reruns more times, then pick the best average.

    This rule gets less noisy evidence instead of reasoning around the noise.
    Averaging m independent runs divides the noise sd by sqrt(m), so the four
    runs of the default (1 original + ``n_reruns=4``) cut it from ~4.75 to ~2.1
    points. Only the ``k`` shortlisted versions are re-run, so each lineage
    costs ``k * n_reruns`` extra feedback runs.

    Each candidate's original score is averaged together with its re-runs. That
    score is biased upward (it is why the version made the shortlist), but it
    is still a real measurement. On simulated data, keeping it did better than
    dropping it at every budget tried. Re-running cannot remove the adaptive
    overfitting that is baked into a version, so regret levels off above zero
    however large the budget.

    Only ``rerun_scores[i]`` for shortlisted versions is read, and at most
    ``n_reruns`` entries of each. Raises RuleUnavailable when a candidate does
    not have that many re-runs.
    """
    scores = np.asarray(feedback_scores, dtype=float)
    shortlist = rerun_shortlist(scores, k)
    averages = []
    for i in shortlist:
        runs = None if rerun_scores is None else rerun_scores[i]
        if runs is None or len(runs) < n_reruns:
            raise RuleUnavailable(f"rerun_top_k needs {n_reruns} re-runs of version H{i}")
        averages.append(np.mean([scores[i], *runs[:n_reruns]]))
    return int(shortlist[_first_argmax(np.array(averages))])


def rerun_shortlist(feedback_scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest feedback scores (earliest on ties), in version order."""
    scores = np.asarray(feedback_scores, dtype=float)
    return np.sort(np.argsort(-scores, kind="stable")[:k])
