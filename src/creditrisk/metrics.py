"""Discrimination, calibration and stability metrics used in PD model validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def gini(y, p) -> float:
    """Accuracy ratio / Gini coefficient = 2*AUC - 1."""
    return 2 * roc_auc_score(y, p) - 1


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov: max distance between bad and good cumulative score distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def hosmer_lemeshow(y, p, n_groups: int = 10) -> tuple[float, float]:
    """Hosmer-Lemeshow chi-square on PD deciles. Returns (statistic, p-value), df = g - 2."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    groups = pd.qcut(pd.Series(p).rank(method="first"), n_groups, labels=False)
    obs = pd.Series(y).groupby(groups).sum()
    exp = pd.Series(p).groupby(groups).sum()
    n = pd.Series(p).groupby(groups).size()
    pbar = exp / n
    chi2 = float((((obs - exp) ** 2) / (n * pbar * (1 - pbar))).sum())
    return chi2, float(stats.chi2.sf(chi2, n_groups - 2))


def summarise(y, p) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    hl, hl_p = hosmer_lemeshow(y, p)
    return {
        "AUC": roc_auc_score(y, p),
        "Gini": gini(y, p),
        "KS": ks_statistic(y, p),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, p),
        "MeanPD": float(np.mean(p)),
        "DefaultRate": float(np.mean(y)),
        "HL_pvalue": hl_p,
    }


def bootstrap_gini(y, preds: dict[str, np.ndarray], reference: str,
                   n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Paired, stratified bootstrap on the test set.

    For every model: 95% CI of Gini, and 95% CI + one-sided p-value of
    (Gini_model - Gini_reference). Paired resampling keeps the correlation between
    models, which is what makes the difference test meaningful.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    names = list(preds)
    draws = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        for j, m in enumerate(names):
            draws[b, j] = gini(y[idx], preds[m][idx])
    ref = draws[:, names.index(reference)]
    rows = []
    for j, m in enumerate(names):
        d = draws[:, j] - ref
        rows.append({
            "model": m,
            "Gini": gini(y, preds[m]),
            "Gini_lo": np.percentile(draws[:, j], 2.5),
            "Gini_hi": np.percentile(draws[:, j], 97.5),
            "dGini_vs_ref": float(np.mean(d)),
            "dGini_lo": np.percentile(d, 2.5),
            "dGini_hi": np.percentile(d, 97.5),
            "p_not_better": float(np.mean(d <= 0)) if m != reference else np.nan,
        })
    return pd.DataFrame(rows)


def psi(expected, actual, n_bins: int = 10) -> float:
    """Population Stability Index on deciles of the expected (development) distribution.
    < 0.1 stable, 0.1-0.25 monitor, > 0.25 significant shift."""
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected), minlength=len(edges) + 1) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual), minlength=len(edges) + 1) / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return df.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), observed_dr=("y", "mean"))
