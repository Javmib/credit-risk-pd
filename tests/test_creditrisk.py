"""Unit tests: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditrisk import capital, metrics, scorecard, stress  # noqa: E402
from creditrisk.data import make_synthetic  # noqa: E402
from creditrisk.features import build_features  # noqa: E402
from creditrisk.models import WoEScorecard, scorecard_coefficients  # noqa: E402
from creditrisk.woe import fit_binning  # noqa: E402

RNG = np.random.default_rng(1)


def test_binning_is_monotonic_and_respects_min_size():
    x = RNG.normal(size=5000)
    y = (RNG.random(5000) < 1 / (1 + np.exp(-(x + 0.3 * np.sin(5 * x))))).astype(int)
    b = fit_binning(x, y, "x", max_bins=12, min_bin_frac=0.05)
    rates = b.table["bad_rate"].to_numpy()
    assert np.all(np.diff(rates) >= 0)
    assert b.table["share"].min() >= 0.05
    assert np.all(np.diff(b.woe) <= 0)          # riskier bins -> lower WoE
    assert b.iv > 0.3


def test_useless_feature_has_tiny_iv():
    x = RNG.normal(size=5000)
    y = (RNG.random(5000) < 0.2).astype(int)
    assert fit_binning(x, y, "noise").iv < 0.02


def test_metrics_on_known_cases():
    y = np.array([0, 0, 1, 1])
    assert metrics.gini(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert metrics.ks_statistic(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    p = RNG.random(2000)
    assert metrics.psi(p, p) < 1e-9
    assert metrics.psi(p, p ** 3) > 0.25


def test_hosmer_lemeshow_accepts_calibrated_model():
    p = RNG.uniform(0.02, 0.6, 20000)
    y = (RNG.random(20000) < p).astype(int)
    _, pval = metrics.hosmer_lemeshow(y, p)
    assert pval > 0.01


def test_score_scaling_anchor_points():
    # 50:1 good:bad odds -> 600 points; doubling odds -> +20 points
    assert abs(scorecard.pd_to_score(1 / 51)[()] - 600) < 1e-6
    assert abs(scorecard.pd_to_score(1 / 101)[()] - 620) < 1e-6


def test_grades_are_ordered():
    g = scorecard.assign_grade(np.array([0.001, 0.03, 0.2, 0.99]))
    assert list(g) == ["G1", "G2", "G5", "G8"]


def test_irb_formula_reference_value():
    # QRRE, PD 1%, LGD 100%: K = N((N^-1(.01) + sqrt(.04) N^-1(.999)) / sqrt(.96)) - .01
    k = capital.irb_capital_k(np.array([0.01]), lgd=1.0)[0]
    from scipy.stats import norm
    expected = norm.cdf((norm.ppf(0.01) + 0.2 * norm.ppf(0.999)) / np.sqrt(0.96)) - 0.01
    assert abs(k - expected) < 1e-12 and abs(k - 0.0306) < 1e-4
    assert capital.irb_capital_k(np.array([1e-6]), 1.0)[0] == capital.irb_capital_k(
        np.array([capital.PD_FLOOR]), 1.0)[0]


def test_scorecard_end_to_end_signs_and_points():
    raw = make_synthetic(6000, seed=3)
    X = build_features(raw)
    sc = WoEScorecard().fit(X, raw["default"])
    intercept, coefs, ok = scorecard_coefficients(sc)
    assert ok and len(coefs) >= 3
    pts = scorecard.points_table(sc.woe_, intercept, coefs)
    # summing points per bin must reproduce the model score (up to rounding)
    Xs = X.iloc[:200]
    score_model = scorecard.pd_to_score(sc.predict_proba(Xs)[:, 1])
    score_points = np.zeros(200)
    for f in coefs:
        per_bin = pts.loc[pts.feature == f, "points"].to_numpy()
        score_points += per_bin[sc.woe_.binnings_[f].bin_index(Xs[f].to_numpy())]
    assert np.max(np.abs(score_points - score_model)) <= len(coefs) * 0.5 + 1e-6


def test_stress_scenarios_raise_pd():
    raw = make_synthetic(6000, seed=4)
    X = build_features(raw)
    sc = WoEScorecard().fit(X, raw["default"])
    out = stress.behavioural_stress(sc, raw, list(X.columns))
    means = [out[s.name].mean() for s in stress.SCENARIOS]
    assert means[0] < means[1] < means[2]
    sysm = stress.systematic_stress(out["Baseline"], capital.RHO_QRRE)
    vals = list(sysm.values())
    np.testing.assert_allclose(vals[0], out["Baseline"])
    assert vals[2].mean() > vals[1].mean() > vals[0].mean()


def test_features_have_no_nans_or_protected_attributes():
    f = build_features(make_synthetic(2000))
    assert not f.isna().any().any()
    assert np.isfinite(f.to_numpy()).all()
    assert "SEX" not in f.columns and "MARRIAGE" not in f.columns


if __name__ == "__main__":  # allows running without pytest
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
