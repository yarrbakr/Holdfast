# Holdfast — Architecture & Project Spec

**Reliable version selection for self-evolving agent harnesses.**

Holdfast is a small research toolkit that tests different rules for choosing the *final* version of a self-evolving agent harness, and measures which rule generalizes best. It runs on simulated data now and is built so that real released data can be dropped in later without changing any logic.

This document is the single source of truth for what to build. A coding agent should read it fully before writing code.

---

## 1. Background — the problem this addresses

This project targets an open problem in the paper **HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?** (ByteDance Seed et al., arXiv:2609.01437).

In the paper's **Evolution** stage, a model starts from a working harness `H0` and repeatedly revises it into new versions `H1, H2, ... Hn`, using feedback from running downstream tasks. During development the model can see each version's score on a **feedback set** (the visible signal). After freezing, each version is also scored on a **held-out set** the model never saw (the ground truth).

At the end, the model must **declare one version as final**. It picks the version with the best feedback score. The paper found this goes wrong:

- The declared version was actually the held-out best in only **2 of 9 lineages**.
- Feedback and held-out scores agreed in direction only about **53% of the time** (barely better than a coin flip).
- Re-running the exact same frozen version swings its score by about **±4.75 points**, so small gains are indistinguishable from noise.

In short: the model optimizes a **noisy, repeatedly-reused validation signal** and often ships a lucky spike that does worse on the tasks that matter. This is a classic model-selection problem (winner's curse, adaptive-holdout overfitting), and the paper's own rule ("pick the highest feedback score") is a weak baseline.

**Holdfast's goal:** build and evaluate better selection rules, and show they beat the max-feedback baseline at lower held-out regret.

---

## 2. What we are building (high-level)

A Python project with a fixed data interface at its core. Everything either *produces* lineages, *selects* a version from a lineage, or *scores* that selection.

The key design principle: **the boundary between the data source and the selection rules is fixed.** The simulator and the future real-data loader return the identical structure, so swapping simulated data for the authors' real data is a one-line change. That is what makes Holdfast "ready for their data."

---

## 3. The core data contract

Every data source must return the same object. Do not deviate from this shape.

A **lineage** is one evolution run: an ordered list of versions `H0 .. Hn`.

Each **version** has:

| Field | Type | Meaning | Who may use it |
|---|---|---|---|
| `feedback_score` | float | Score on the visible feedback set during development | Selection rules **may** use this |
| `heldout_score` | float | Score on the disjoint held-out set (ground truth) | **Evaluation only — never a selection rule** |
| `features` | dict | Observable metadata about the edit | Selection rules may use this |

`features` contains:

- `edit_files` (int) — number of files changed vs the previous version
- `edit_lines` (int) — net lines changed
- `touched_verification` (bool) — did the edit modify the verifier/checking logic
- `revision_calls` (int) — number of revise-and-recheck cycles the model made

**Hard rule:** a selection rule must never read `heldout_score`. That field exists only so the scorer can grade the rule's choice after the fact. This is enforced by the rule function signatures and by a unit test.

---

## 4. Components and how they relate

```
                         ┌─────────────────────┐
   data/simulator.py ───▶│                     │
   (fake lineages)       │  list of lineages   │
                         │  (the data contract)│
   data/loader.py  ─────▶│                     │
   (real lineages,       └──────────┬──────────┘
    plugged in later)               │
                                     ▼
                          ┌──────────────────────┐
                          │  rules/selection.py  │   each rule sees only
                          │  rule(visible) → idx │   feedback_score + features
                          └──────────┬───────────┘
                                     │ chosen version index
                                     ▼
                          ┌──────────────────────┐
                          │      metrics.py      │   uses heldout_score
                          │  held_out_regret     │   to grade the choice
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │      evaluate.py     │   leave-one-lineage-out
                          │  run rules over all  │   for any tunable rule
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │        run.py        │   prints table + saves
                          │  table + 2 figures   │   figures/
                          └──────────────────────┘
```

**Component responsibilities:**

1. **`data/simulator.py`** — generates realistic fake lineages (see §6). Seed-controlled.
2. **`data/loader.py`** — reads real lineages from CSV/JSON into the same structure. Ships as a documented stub with the parsing skeleton in place, so real data can be added by filling one function.
3. **`rules/selection.py`** — the selection rules (see §5). Each is a pure function.
4. **`metrics.py`** — held-out regret and win-rate vs baseline.
5. **`evaluate.py`** — runs each rule across all lineages; uses leave-one-lineage-out for any rule with a tunable knob so it is never tested on a lineage it was tuned on.
6. **`run.py`** — single entrypoint; prints the results table and saves the two figures.
7. **`tests/`** — correctness tests (see §8).
8. **`README.md`** — problem, rules, how to run, how to plug in real data, and the honest caveat.

---

## 5. The selection rules

Each rule has the signature `rule(feedback_scores, features) -> chosen_index`. None of them receive `heldout_score`.

**Baselines (the floor to beat):**

- `baseline_max_feedback` — pick the version with the highest feedback score. This is the paper's rule and the target to beat.
- `pick_last` — always pick the final version `Hn`.
- `pick_h0` — always pick `H0` (do nothing).

**Contenders:**

- `one_standard_error` — estimate the noise (standard error) of a feedback score; among all versions within one SE of the best score, pick the **earliest** one. Rationale: when several versions are statistically tied, prefer the earlier, less-overfit version.
- `thresholdout` — walk versions in order keeping an incumbent; switch to a new version only if its feedback gain over the incumbent exceeds a noise threshold. Ignores lucky spikes.
- `validation_split` — hold back part of the feedback signal purely for selection (never used to drive the choice otherwise); pick the version best on that untouched split. Requires a per-task feedback interface, so it is enabled when per-task data is available.
- `shrinkage` — shrink each version's feedback score toward the lineage mean by an amount set by the noise (empirical-Bayes / James-Stein style), then pick the argmax of the shrunk scores. Corrects the winner's curse directly.

Rules must be easy to add: one new function is all it should take.

---

## 6. The simulator (matching the paper)

`data/simulator.py` generates **9 lineages** whose statistics mirror the paper:

- **Non-monotonic** feedback trajectories (scores rise and fall across versions, not a steady climb).
- Feedback noise with **standard deviation ≈ 4.75 points**.
- A latent "true quality" per version that drifts slowly; `heldout_score` is a cleaner readout of true quality, `feedback_score` is true quality plus the 4.75-point noise.
- Tuned so that **feedback and held-out scores agree in direction only about 50% of the time**, reproducing the paper's core difficulty.
- Version-chain length varied per lineage (roughly 5–10 versions), like the real trajectories.
- `features` correlated with true quality in the way the paper reports (e.g. more `revision_calls` and `touched_verification=True` weakly predict a real, held-out improvement).

Everything seed-controlled so results are exactly reproducible.

---

## 7. Metric and evaluation

**Held-out regret** for one lineage:

```
regret = heldout_score[oracle_best_index] - heldout_score[chosen_index]
```

where `oracle_best_index` is the version with the highest `heldout_score`. Regret is 0 when the rule picks the truly best version, and positive otherwise.

Reported per rule:

- **Mean held-out regret** across all lineages (lower is better).
- **Win rate vs baseline** — fraction of lineages where the rule's regret is strictly lower than `baseline_max_feedback`.

**Leave-one-lineage-out:** any rule with a tunable knob (e.g. the threshold in `thresholdout`) is tuned on 8 lineages and tested on the 9th, rotated across all lineages, so a rule is never graded on data used to tune it. This guards against the selector itself overfitting the small sample.

---

## 8. Tests

`tests/` must include at least:

- **No peeking:** assert that no selection rule accesses `heldout_score` (enforce via signature / a guard object).
- **Zero regret on oracle:** a rule that returns `oracle_best_index` yields regret 0.
- **Contract check:** simulator output conforms exactly to the §3 data contract.
- **Determinism:** a fixed seed reproduces identical lineages and identical results.

---

## 9. Expected results (when we run it)

These are the outcomes we expect on the simulated data. They are **illustrative hypotheses**, not claims about the authors' real system, and the README must say so.

- `baseline_max_feedback` will show clearly **positive** mean regret — it frequently ships a lucky spike, matching the paper's finding.
- `one_standard_error` and `thresholdout` are expected to **reduce mean regret** and win on a majority of lineages, because they refuse to chase gains inside the noise band.
- `shrinkage` should land in a similar place, correcting the winner's curse in a single formula.
- `validation_split` (once per-task data exists) is expected to be the strongest, since it selects on an honest signal.
- `pick_h0` and `pick_last` are sanity floors; they should not beat the good rules.

The deliverable proves three things: the problem is defined precisely, the evaluation is correct and honest (leave-one-lineage-out, no peeking), and the statistical rules the literature recommends actually help. When real lineages are loaded, the same code produces the real result.

**Figures produced:**

1. `figures/trajectory.png` — feedback vs held-out score across versions for one sample lineage, showing the model chasing the feedback peak while the held-out best sits elsewhere.
2. `figures/regret.png` — bar chart of mean held-out regret per rule.

---

## 10. Plugging in real data

When the authors share their data, only `data/loader.py` changes:

- Their per-version feedback and held-out scores map onto `feedback_score` / `heldout_score`.
- Their harness diffs map onto `features`.
- If per-task feedback results are available, they enable `validation_split`.

`run.py` gains a flag to choose the data source (`--source sim` or `--source real`). Nothing else changes, because both sources honor the §3 contract.

---

## 11. Constraints

- Python 3.11+. Free, open-source libraries only: `numpy`, `pandas`, `matplotlib`, and `scikit-learn` (only if a learned selector is added later). **No LLM APIs, no paid services, no API keys anywhere.**
- Fully reproducible: fixed seeds, versions pinned in `requirements.txt`.
- Clean, small, well-commented functions with clear names.
- Runs end to end with: `pip install -r requirements.txt && python run.py`.
