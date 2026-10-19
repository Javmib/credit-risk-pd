"""Score scaling, points table and a PD master scale (rating grades)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

# Score = OFFSET + FACTOR * ln(good:bad odds). 600 points at 50:1 odds, 20 points to double.
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def pd_to_score(pd_: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(pd_, float), 1e-6, 1 - 1e-6)
    return OFFSET + FACTOR * np.log((1 - p) / p)


def points_table(woe_transformer, intercept: float, coefs: dict) -> pd.DataFrame:
    """Integer points per bin. Total score = sum of points (+ rounding error)."""
    n = len(coefs)
    rows = []
    for feat, beta in coefs.items():
        b = woe_transformer.binnings_[feat]
        for bin_label, w in zip(b.table["bin"], b.woe):
            # log-odds(bad) = intercept + sum beta*woe ; score uses log-odds(good)
            pts = -(beta * w + intercept / n) * FACTOR + OFFSET / n
            rows.append({"feature": feat, "bin": bin_label, "woe": w, "points": int(round(pts))})
    return pd.DataFrame(rows)


# Master scale: PD upper bounds per grade (a simplified, bank-style 8-grade scale).
GRADE_BOUNDS = [0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.0]
GRADE_LABELS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]


def assign_grade(pd_: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(GRADE_BOUNDS, np.asarray(pd_), side="left")
    return np.array(GRADE_LABELS)[np.minimum(idx, len(GRADE_LABELS) - 1)]


def grade_backtest(y, pd_) -> pd.DataFrame:
    """Per grade: count, mean PD, observed default rate, and a two-sided binomial test of
    H0: true default rate = mean PD. Low p-values flag mis-calibrated grades."""
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_), "grade": assign_grade(pd_)})
    rows = []
    for g in GRADE_LABELS:
        s = df[df["grade"] == g]
        if len(s) == 0:
            continue
        k, n, p = int(s["y"].sum()), len(s), float(s["pd"].mean())
        rows.append({
            "grade": g, "n": n, "share": n / len(df), "mean_pd": p,
            "observed_dr": k / n, "defaults": k,
            "binom_p": binomtest(k, n, p).pvalue,
        })
    return pd.DataFrame(rows)
