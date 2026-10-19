"""Behavioural feature engineering.

Features are the kind used in retail behavioural scorecards: utilisation, repayment
coverage, delinquency depth/frequency and their trends. Protected characteristics
(SEX, MARRIAGE) are deliberately excluded from model inputs; they are kept in the
raw data only so that the model can be checked for disparate outcomes afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BILL_COLS, MONTHS, PAY_COLS, PAYAMT_COLS

EXCLUDED_PROTECTED = ["SEX", "MARRIAGE"]


def _safe_div(a, b, fill: float):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, fill, dtype=float)
    mask = b > 0
    out[mask] = a[mask] / b[mask]
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    limit = df["LIMIT_BAL"].to_numpy(float)
    bills = df[BILL_COLS].to_numpy(float)
    pays = df[PAY_COLS].to_numpy(float)
    payamts = df[PAYAMT_COLS].to_numpy(float)

    util = np.clip(bills / limit[:, None], -0.5, 2.0)  # column j = month j+1

    # Payment in month k settles the bill issued in month k+1 (one month earlier).
    # No positive bill outstanding -> treated as fully covered.
    cover = np.column_stack([
        np.clip(_safe_div(payamts[:, k], bills[:, k + 1], fill=1.0), 0, 2)
        for k in range(len(MONTHS) - 1)
    ])

    months_late = np.clip(pays, 0, None)

    f = pd.DataFrame(index=df.index)
    f["log_limit"] = np.log(limit)
    f["age"] = df["AGE"].astype(float)
    f["education"] = df["EDUCATION"].astype(float)
    # Most recent repayment status: the single strongest behavioural driver.
    f["status_m1"] = pays[:, 0]
    f["status_m2"] = pays[:, 1]
    f["max_months_late_6m"] = months_late.max(axis=1)
    f["n_months_late_6m"] = (months_late > 0).sum(axis=1)
    f["n_months_late_3m"] = (months_late[:, :3] > 0).sum(axis=1)
    f["n_paid_in_full_6m"] = (pays == -1).sum(axis=1)
    f["util_m1"] = util[:, 0]
    f["util_avg_6m"] = util.mean(axis=1)
    f["util_max_6m"] = util.max(axis=1)
    f["util_trend_6m"] = util[:, 0] - util[:, 5]
    f["cover_m1"] = cover[:, 0]
    f["cover_avg_5m"] = cover.mean(axis=1)
    f["cover_min_5m"] = cover.min(axis=1)
    f["pay_to_limit_6m"] = payamts.sum(axis=1) / limit
    f["bill_volatility_6m"] = bills.std(axis=1) / limit
    f["headroom_m1"] = np.clip((limit - bills[:, 0]) / limit, -1, 1.5)
    return f


# Expected direction of each feature's effect on default risk (+1 riskier as it grows).
# Used as monotonic constraints for the constrained GBM, and as sign checks for the scorecard.
MONOTONE = {
    "log_limit": -1,
    "status_m1": +1,
    "status_m2": +1,
    "max_months_late_6m": +1,
    "n_months_late_6m": +1,
    "n_months_late_3m": +1,
    "n_paid_in_full_6m": -1,
    "util_m1": +1,
    "util_avg_6m": +1,
    "util_max_6m": +1,
    "cover_m1": -1,
    "cover_avg_5m": -1,
    "cover_min_5m": -1,
    "pay_to_limit_6m": -1,
    "headroom_m1": -1,
}
