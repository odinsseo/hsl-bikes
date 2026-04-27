# Statistical inference for research questions (RQ1–RQ3)

This document is the canonical reference for **what the hypothesis tests assume**, how **dependence** (time and space) affects interpretation, what we **can and cannot** claim, and how this ties to code (`paired_sign_permutation_pvalue`, `bootstrap_mean_ci` in `scripts/experiments/train_eval.py`, and `scripts/experiments/rq_hypothesis_tests.py`).

## Estimand

- **Primary quantity**: mean over stations of **paired differences** in **per-station test WMAPE** between two configurations (e.g. two graph sets, or station vs broadcast-community predictions), on the **same** test window and **same** station identifiers.
- Each per-station WMAPE is a **scalar summary** over the full test horizon for that station (as implemented by `station_wmape_vector`).

## What we are not doing (time permutation)

- We do **not** permute individual time indices. Classical permutation tests for **time series** often assume **exchangeability** under the null; serial dependence generally breaks exchangeability, so naive time permutations need not control Type I error at nominal α (see Romano & Wolf, [Permutation Testing for Dependence in Time Series](https://arxiv.org/abs/2009.03170)).
- The **paired sign-randomization** test randomizes **signs of paired station-level differences** \(D_s = \mathrm{WMAPE}_s^{(A)} - \mathrm{WMAPE}_s^{(B)}\). That is a different randomization hypothesis from “time points are exchangeable.”

## Temporal dependence (within each station)

- Errors at different times within a station are **autocorrelated**. That affects how informative each station’s scalar WMAPE is (effective number of time steps is less than the horizon length).
- Interpretation: we infer about **performance on this fixed evaluation window**, not about i.i.d. replicates of calendar days unless the design is extended (e.g. rolling origins, multiple blocks).

## Spatial / cross-station dependence

- Stations are **not independent** (shared weather, geography, demand).
- **Paired sign tests** rely on symmetry arguments on \(D_s\) across stations \(s\). Strong spatial correlation reduces **effective sample size** relative to the number of stations; nominal **p-values** should be read as **approximate** under departure from ideal symmetry/independence.
- **Bootstrap confidence intervals** that **resample stations with replacement** (`bootstrap_mean_ci` on the vector of station-level differences) assume approximate independence across resampled units. Under **spatial dependence**, nominal **CI coverage can be miscalibrated** (often optimistic when correlation is positive). Literature points to **block bootstrap** along time or **spatial block** methods (e.g. Lahiri et al., [Annals of Statistics 2006](https://projecteuclid.org/journals/annals-of-statistics/volume-34/issue-4/Resampling-methods-for-spatial-regression-models-under-a-class-of/10.1214/009053606000000551.pdf); moving block / stationary bootstrap for persistent series, e.g. [Mudelsee](https://manfredmudelsee.com/publ/pdf/Estimating_Pearson_correlation_coefficient_with_bootstrap_conf_interval_from_serially_dependent_time_series.pdf)).
- **Reporting**: always report **effect size** and **n_stations_used**; treat bootstrap CIs as **supporting** unless block or cluster resampling is added.

## Multiplicity (Holm–Bonferroni)

- **Holm** adjustment is applied **within each (RQ, cohort)** family of contrasts. It is **valid under arbitrary dependence** among the constituent tests (it may be conservative).

## RQ3: community forecasts broadcast to stations

- Community-level `graph_propagation` predictions are **broadcast** to each station column (each station receives its community’s predicted series) and evaluated against **station** actuals. The paired comparison with the **station-aggregation** run is well defined on station index.
- This estimand is **not** the same as comparing errors in **native community outcome space** (different dimensions). State explicitly which definition you use in any write-up.

## Direction and H₀

- **H₀**: mean paired difference in per-station test WMAPE equals zero for the stated contrast (see `rq_hypothesis_tests.csv` column `H0`).
- The implementation uses the existing **two-sided** paired sign permutation on the mean difference (see `paired_sign_permutation_pvalue` in `train_eval.py`). **Reject H₀** when Holm-adjusted *p* \< α (default 0.05).

## What we can claim vs what needs extra work

**Reasonable claims (single held-out split, after `rq_runner` + `rq_hypothesis_tests`):**

- On **this** protocol and **this** test window, whether mean paired station-level WMAPE **differs** between configurations A and B, with multiplicity control within each (RQ, cohort) family.

**Not implied without stronger designs:**

- Broad **temporal generalization** beyond the chosen test period.
- **Causal** interpretation of graph construction.
- **Tightly calibrated** uncertainty under spatial dependence without block/cluster bootstrap or hierarchical modeling.

## Implementation efficiency

- One length-\(S\) vector per experiment is small compared to full prediction tensors.
- Permutation is batched in `train_eval`; cost is \(O(B \cdot S)\) for \(B\) randomizations—negligible vs model fitting.

## Revision log

- Update this file when you change α, the Holm family definition, cohort policy (`sparse_quantile`), or the broadcast rule for RQ3 so thesis text stays aligned with artifacts.
