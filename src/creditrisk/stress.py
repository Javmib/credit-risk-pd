"""Stress testing the PD model.

Two complementary approaches:
1. Behavioural scenario shocks: perturb the account-level drivers (balances, repayments,
   arrears roll-rates), rebuild features and re-score with the fitted model. This tests
   how the model propagates a deterioration in customer behaviour.
2. Systematic-factor shift (Vasicek one-factor model): condition the baseline PDs on an
   adverse realisation of the macro factor Z, the same mechanism that links macro
   scenarios to PDs in IFRS 9 / regulatory stress tests.
Scenario severities are illustrative, not calibrated to a published macro scenario.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .capital import vasicek_conditional_pd
from .data import BILL_COLS, PAYAMT_COLS
from .features import build_features


@dataclass(frozen=True)
class Scenario:
    name: str
    balance_mult: float      # balances grow as customers draw down lines
    repayment_mult: float    # repayments shrink
    roll_to_arrears: float   # share of up-to-date accounts that roll into 1 month arrears
    deepen_arrears: int      # extra months added to accounts already in arrears


SCENARIOS = [
    Scenario("Baseline", 1.00, 1.00, 0.00, 0),
    Scenario("Adverse", 1.10, 0.85, 0.05, 0),
    Scenario("Severely adverse", 1.20, 0.70, 0.12, 1),
]

FACTOR_SCENARIOS = {"Baseline (Z = 0)": 0.0, "1-in-10 (Z = -1.28)": norm.ppf(0.10),
                    "1-in-25 (Z = -1.75)": norm.ppf(0.04)}


def apply_scenario(raw: pd.DataFrame, s: Scenario, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = raw.copy()
    df[BILL_COLS[:2]] = df[BILL_COLS[:2]] * s.balance_mult   # last two statements
    df[PAYAMT_COLS[:2]] = df[PAYAMT_COLS[:2]] * s.repayment_mult
    current = df["PAY_1"] <= 0
    roll = current & (rng.random(len(df)) < s.roll_to_arrears)
    df.loc[roll, "PAY_1"] = 1
    late = (df["PAY_1"] > 0) & ~roll
    df.loc[late, "PAY_1"] = np.minimum(df.loc[late, "PAY_1"] + s.deepen_arrears, 9)
    return df


def behavioural_stress(model, raw: pd.DataFrame, feature_cols, scenarios=SCENARIOS):
    out = {}
    for s in scenarios:
        X = build_features(apply_scenario(raw, s))[feature_cols]
        out[s.name] = model.predict_proba(X)[:, 1]
    return out


def systematic_stress(pd_base: np.ndarray, rho: float, scenarios=FACTOR_SCENARIOS):
    """Shift PDs by conditioning on Z, centred so that Z = 0 reproduces the mean PD.

    The plain Vasicek formula at Z = 0 gives the *median* conditional PD, which is below
    the unconditional mean; we rescale the shift so the baseline is unchanged."""
    out = {}
    for name, z in scenarios.items():
        cond = vasicek_conditional_pd(pd_base, z, rho)
        base = vasicek_conditional_pd(pd_base, 0.0, rho)
        # ratio of conditional to median-conditional PD, applied to the baseline PD
        out[name] = np.clip(pd_base * cond / base, 0, 1)
    return out
