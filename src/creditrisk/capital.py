"""From PD to expected loss and Basel IRB capital for a revolving retail book.

Illustrative only: LGD and CCF are not modelled from data here (the dataset has no
recovery or drawdown history), they are stated assumptions passed as parameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Asset correlation for qualifying revolving retail exposures (Basel framework, CRE31).
RHO_QRRE = 0.04
PD_FLOOR = 0.001   # 0.10% floor for QRRE revolvers under Basel 3.1
CONFIDENCE = 0.999


def ead(limit, drawn, ccf: float = 0.5) -> np.ndarray:
    """EAD = drawn balance + CCF x undrawn headroom (negative balances treated as 0)."""
    drawn = np.clip(np.asarray(drawn, float), 0, None)
    undrawn = np.clip(np.asarray(limit, float) - drawn, 0, None)
    return drawn + ccf * undrawn


def vasicek_conditional_pd(pd_, z, rho: float = RHO_QRRE) -> np.ndarray:
    """PD conditional on the systematic factor Z = z (z < 0 is a downturn)."""
    pd_ = np.clip(np.asarray(pd_, float), 1e-9, 1 - 1e-9)
    return norm.cdf((norm.ppf(pd_) - np.sqrt(rho) * z) / np.sqrt(1 - rho))


def irb_capital_k(pd_, lgd: float, rho: float = RHO_QRRE) -> np.ndarray:
    """Capital requirement K per unit EAD (retail: no maturity adjustment)."""
    p = np.maximum(np.asarray(pd_, float), PD_FLOOR)
    stressed = vasicek_conditional_pd(p, norm.ppf(1 - CONFIDENCE), rho)
    return lgd * stressed - lgd * p


def portfolio_summary(pd_, exposure, lgd: float) -> dict:
    pd_ = np.asarray(pd_, float)
    exposure = np.asarray(exposure, float)
    k = irb_capital_k(pd_, lgd)
    el = pd_ * lgd * exposure
    return {
        "EAD": exposure.sum(),
        "EAD-weighted PD": float(np.average(pd_, weights=exposure)),
        "Expected loss": el.sum(),
        "EL / EAD": el.sum() / exposure.sum(),
        "Capital K": float((k * exposure).sum()),
        "RWA": float(12.5 * (k * exposure).sum()),
        "RWA density": float(12.5 * (k * exposure).sum() / exposure.sum()),
    }


def summary_frame(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).T
