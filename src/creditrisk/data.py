"""Data acquisition and cleaning for the UCI "Default of Credit Card Clients" dataset.

Source: Yeh, I-C. & Lien, C-H. (2009). The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients. Expert Systems with
Applications 36(2), 2473-2480. UCI Machine Learning Repository, dataset id 350 (CC BY 4.0).

30,000 credit-card accounts of a Taiwanese bank, observed April-September 2005.
Target: default on the payment due in October 2005 (1 = default).
Monthly columns are indexed 1..6 where 1 = September 2005 (most recent) and 6 = April 2005.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
TARGET = "default"
MONTHS = range(1, 7)

PAY_COLS = [f"PAY_{k}" for k in MONTHS]
BILL_COLS = [f"BILL_AMT{k}" for k in MONTHS]
PAYAMT_COLS = [f"PAY_AMT{k}" for k in MONTHS]
RAW_COLS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_COLS + BILL_COLS + PAYAMT_COLS + [TARGET]
)


def download(raw_dir: str | Path = "data/raw") -> Path:
    """Download and unzip the UCI file. Returns the path to the .xls file."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = find_raw_file(raw_dir)
    if existing is not None:
        return existing
    print(f"Downloading {UCI_URL} ...")
    with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(raw_dir)
    path = find_raw_file(raw_dir)
    if path is None:
        raise FileNotFoundError(f"No data file found in {raw_dir} after extraction")
    return path


def find_raw_file(raw_dir: str | Path) -> Path | None:
    """Accept the UCI .xls or the Kaggle mirror CSV (UCI_Credit_Card.csv)."""
    raw_dir = Path(raw_dir)
    for pattern in ("*.xls", "*.xlsx", "*.csv"):
        hits = sorted(raw_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(path, header=1)  # first row is an "X1..X23" label row
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={
        "default payment next month": TARGET,
        "default.payment.next.month": TARGET,
        "PAY_0": "PAY_1",  # the original file skips PAY_1; harmonise the index
    })
    missing = set(RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {sorted(missing)}")
    return df[RAW_COLS].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix undocumented category codes; enforce dtypes."""
    df = df.copy()
    # EDUCATION: 1 grad school, 2 university, 3 high school, 4 others. 0/5/6 are undocumented.
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4)
    # MARRIAGE: 1 married, 2 single, 3 others. 0 is undocumented.
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2]), 3)
    # PAY_k: -2 no consumption, -1 paid in full, 0 revolving (min. paid), 1..9 months late.
    for c in PAY_COLS:
        df[c] = df[c].clip(lower=-2, upper=9)
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def load(raw_dir: str | Path = "data/raw", allow_download: bool = True) -> pd.DataFrame:
    path = find_raw_file(raw_dir)
    if path is None:
        if not allow_download:
            raise FileNotFoundError(f"No data in {raw_dir}")
        path = download(raw_dir)
    return clean(load_raw(path))


def make_synthetic(n: int = 30_000, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the same schema, for tests and offline smoke runs ONLY.

    A latent creditworthiness factor drives limits, utilisation, repayment behaviour,
    delinquency and default, so the pipeline has realistic structure to find.
    Numbers produced from this data must never be reported as results.
    """
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)  # higher = riskier
    limit = np.round(np.exp(11.5 - 0.45 * risk + rng.normal(0, 0.6, n)), -4).clip(10_000, 1_000_000)
    age = rng.integers(21, 75, n)
    edu = rng.choice([1, 2, 3, 4], n, p=[0.35, 0.47, 0.16, 0.02])
    df = pd.DataFrame({
        "LIMIT_BAL": limit, "SEX": rng.choice([1, 2], n), "EDUCATION": edu,
        "MARRIAGE": rng.choice([1, 2, 3], n, p=[0.45, 0.53, 0.02]), "AGE": age,
    })
    util_base = 1 / (1 + np.exp(-(0.2 + 0.9 * risk + rng.normal(0, 0.8, n))))
    for k in MONTHS:  # k=1 most recent; risk drifts upward toward the present
        drift = 0.08 * (6 - k) * (risk > 0.8)
        u = np.clip(util_base + drift + rng.normal(0, 0.08, n), -0.05, 1.3)
        bill = np.round(u * limit)
        df[f"BILL_AMT{k}"] = bill
        late = 0.5 * risk + drift * 3 + rng.normal(0, 0.9, n)
        df[f"PAY_{k}"] = np.select(
            [late < -1.2, late < -0.4, late < 0.9, late < 1.6, late < 2.2],
            [-2, -1, 0, 1, 2], default=3)
    for k in MONTHS:
        prev_bill = df[f"BILL_AMT{min(k + 1, 6)}"].to_numpy()
        ratio = np.clip(1 / (1 + np.exp(-(-0.3 - 1.1 * risk + rng.normal(0, 1, n)))), 0, 1)
        df[f"PAY_AMT{k}"] = np.round(np.maximum(prev_bill, 0) * ratio)
    logit = -1.55 + 0.85 * risk + 0.35 * (df["PAY_1"] > 0) + rng.normal(0, 0.5, n)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return clean(df[RAW_COLS])
