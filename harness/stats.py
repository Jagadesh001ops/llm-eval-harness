"""Uncertainty estimates for per-item eval scores.

Why item-level bootstrap: every stored run uses temperature 0, and repeated runs of the same config are
(near-)identical (see analysis/run_determinism.py), so re-running a model says almost nothing about
uncertainty. The dominant source of uncertainty is which test items happened to be sampled, so we
resample items with replacement.

Conventions (also in docs/statistics.md):
  * B = 10,000 resamples, percentile intervals, alpha = 0.05 (95% CI).
  * Every call re-seeds numpy's PCG64 with DEFAULT_SEED, so a given (n, B, seed) always uses the same
    resample indices: results are reproducible and independent of the order cells are processed in.
  * Paired comparisons (same items, two systems) resample item indices jointly, which keeps the
    per-item correlation between systems; this is much tighter than comparing two separate CIs.
"""

import numpy as np
from scipy import stats as _st

DEFAULT_B = 10_000
DEFAULT_SEED = 20261118
ALPHA = 0.05


def _indices(n, B, seed):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def bootstrap_mean_ci(values, B=DEFAULT_B, seed=DEFAULT_SEED, alpha=ALPHA):
    """Percentile bootstrap CI for the mean of per-item scores (0/1 accuracy, ROUGE-L, ...)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    boots = x[_indices(n, B, seed)].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": n}


def bootstrap_ratio_ci(numer, denom, B=DEFAULT_B, seed=DEFAULT_SEED, alpha=ALPHA):
    """CI for sum(numer)/sum(denom) resampling items, e.g. accuracy among items that got a valid label."""
    num = np.asarray(numer, dtype=float)
    den = np.asarray(denom, dtype=float)
    n = len(num)
    if n == 0 or den.sum() == 0:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "n": int(den.sum()) if n else 0}
    idx = _indices(n, B, seed)
    d = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = num[idx].sum(axis=1) / d
    boots = boots[d > 0]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": float(num.sum() / den.sum()), "lo": float(lo), "hi": float(hi), "n": int(den.sum())}


def paired_bootstrap_diff(a, b, B=DEFAULT_B, seed=DEFAULT_SEED, alpha=ALPHA):
    """CI for mean(a) - mean(b) where a[i] and b[i] are scores of two systems on the same item i."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same items in the same order")
    d = a - b
    boots = d[_indices(len(d), B, seed)].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(d)}


def mcnemar_exact(a, b):
    """Exact two-sided McNemar test for paired binary outcomes.
    Returns (n_a_only, n_b_only, p) where n_a_only = items a got right and b got wrong."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    k = a_only + b_only
    p = 1.0 if k == 0 else float(_st.binomtest(a_only, k, 0.5, alternative="two-sided").pvalue)
    return a_only, b_only, p


def wilson_ci(k, n, alpha=ALPHA):
    """Wilson score interval for a binomial proportion (analytic cross-check on the bootstrap)."""
    if n == 0:
        return (np.nan, np.nan)
    ci = _st.binomtest(int(k), int(n)).proportion_ci(confidence_level=1 - alpha, method="wilson")
    return (float(ci.low), float(ci.high))


def holm_adjust(pvalues):
    """Holm-Bonferroni adjusted p-values (controls family-wise error across a set of comparisons)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
