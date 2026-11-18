import numpy as np
import pytest

from harness.stats import (bootstrap_mean_ci, bootstrap_ratio_ci, holm_adjust, mcnemar_exact,
                           paired_bootstrap_diff, wilson_ci)


def test_bootstrap_is_deterministic_and_brackets_point():
    x = np.r_[np.ones(81), np.zeros(19)]
    r1, r2 = bootstrap_mean_ci(x), bootstrap_mean_ci(x)
    assert r1 == r2
    assert r1["point"] == pytest.approx(0.81)
    assert r1["lo"] < r1["point"] < r1["hi"]


def test_bootstrap_close_to_wilson_at_n100():
    # For a mid-range proportion at n=100 the percentile bootstrap and Wilson should agree to ~1.5 pts.
    x = np.r_[np.ones(81), np.zeros(19)]
    b = bootstrap_mean_ci(x)
    w = wilson_ci(81, 100)
    assert abs(b["lo"] - w[0]) < 0.015 and abs(b["hi"] - w[1]) < 0.015


def test_bootstrap_coverage_simulation():
    # Empirical coverage of the 95% interval for Bernoulli(0.8), n=100 should be near 95%.
    rng = np.random.default_rng(0)
    hits = 0
    sims = 300
    for s in range(sims):
        x = rng.random(100) < 0.8
        r = bootstrap_mean_ci(x, B=2000, seed=s)
        hits += r["lo"] <= 0.8 <= r["hi"]
    assert 0.90 <= hits / sims <= 0.99


def test_paired_diff_tighter_than_unpaired_when_correlated():
    rng = np.random.default_rng(1)
    base = (rng.random(200) < 0.8).astype(float)
    a = base.copy()
    b = base.copy()
    flip = rng.choice(200, 10, replace=False)
    b[flip] = 0  # b is a on every item except 10 extra misses
    d = paired_bootstrap_diff(a, b)
    width_paired = d["hi"] - d["lo"]
    ca, cb = bootstrap_mean_ci(a), bootstrap_mean_ci(b)
    assert width_paired < (ca["hi"] - ca["lo"])
    assert d["lo"] > 0 or a[flip].sum() == 0


def test_mcnemar_matches_hand_computation():
    # 12 discordant pairs all in one direction: p = 2 * 0.5**12
    a = np.r_[np.ones(12), np.ones(83), np.zeros(5)]
    b = np.r_[np.zeros(12), np.ones(83), np.zeros(5)]
    a_only, b_only, p = mcnemar_exact(a, b)
    assert (a_only, b_only) == (12, 0)
    assert p == pytest.approx(2 * 0.5 ** 12)


def test_ratio_ci_point_is_selective_accuracy():
    correct = np.r_[np.ones(70), np.zeros(30)]
    valid = np.r_[np.ones(90), np.zeros(10)]
    r = bootstrap_ratio_ci(correct * valid, valid)
    assert r["point"] == pytest.approx(70 / 90)
    assert r["n"] == 90


def test_holm():
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
