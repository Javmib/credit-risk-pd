"""Weight-of-Evidence binning, Information Value and feature selection.

Implemented from scratch (no scorecard library) as a scikit-learn transformer, so binning
is fitted inside each training fold and never sees validation or test data.

Conventions (Siddiqi, "Credit Risk Scorecards", 2006):
    WoE_i = ln( %goods_i / %bads_i )      positive WoE = lower risk than average
    IV    = sum_i (%goods_i - %bads_i) * WoE_i
Rule of thumb: IV < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, > 0.3 strong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class FeatureBinning:
    name: str
    edges: np.ndarray            # interior cut points; bin i = (edges[i-1], edges[i]]
    woe: np.ndarray
    table: pd.DataFrame = field(repr=False)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def bin_index(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, x, side="left")

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.woe[self.bin_index(x)]


def _counts(idx: np.ndarray, y: np.ndarray, n_bins: int):
    bads = np.bincount(idx, weights=y, minlength=n_bins)
    tot = np.bincount(idx, minlength=n_bins).astype(float)
    return tot - bads, bads


def _woe_iv(goods, bads, smoothing: float = 0.5):
    g = goods + smoothing
    b = bads + smoothing
    pg, pb = g / g.sum(), b / b.sum()
    woe = np.log(pg / pb)
    return woe, (pg - pb) * woe


def fit_binning(x: np.ndarray, y: np.ndarray, name: str, max_bins: int = 10,
                min_bin_frac: float = 0.05, monotonic: bool = True) -> FeatureBinning:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    qs = np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1])
    edges = np.unique(qs)
    edges = edges[edges < x.max()]  # a cut at the max would leave an empty last bin

    def stats(e):
        idx = np.searchsorted(e, x, side="left")
        return _counts(idx, y, len(e) + 1)

    # 1) Merge bins that are too small or contain no goods / no bads.
    n_min = min_bin_frac * len(x)
    while len(edges) > 0:
        goods, bads = stats(edges)
        tot = goods + bads
        bad = np.where((tot < n_min) | (goods == 0) | (bads == 0))[0]
        if len(bad) == 0:
            break
        i = bad[np.argmin(tot[bad])]
        # merge with the smaller neighbour (remove the edge between them)
        if i == 0:
            j = 0
        elif i == len(edges):
            j = i - 1
        else:
            j = i - 1 if tot[i - 1] <= tot[i + 1] else i
        edges = np.delete(edges, j)

    # 2) Enforce a monotonic bad rate in the direction of the overall trend.
    if monotonic and len(edges) > 0:
        rank_corr = pd.Series(x).rank().corr(pd.Series(y))
        sign = 1.0 if (np.isnan(rank_corr) or rank_corr >= 0) else -1.0
        while len(edges) > 0:
            goods, bads = stats(edges)
            rate = bads / (goods + bads)
            diffs = sign * np.diff(rate)
            viol = np.where(diffs < 0)[0]
            if len(viol) == 0:
                break
            j = viol[np.argmin(diffs[viol])]  # worst reversal: merge bins j and j+1
            edges = np.delete(edges, j)

    goods, bads = stats(edges)
    woe, iv = _woe_iv(goods, bads)
    lo = np.r_[-np.inf, edges]
    hi = np.r_[edges, np.inf]
    table = pd.DataFrame({
        "feature": name,
        "bin": [f"({a:.4g}, {b:.4g}]" for a, b in zip(lo, hi)],
        "n": (goods + bads).astype(int),
        "share": (goods + bads) / len(x),
        "bads": bads.astype(int),
        "bad_rate": bads / (goods + bads),
        "woe": woe,
        "iv": iv,
    })
    return FeatureBinning(name, edges, woe, table)


class WoETransformer(BaseEstimator, TransformerMixin):
    """Fit WoE bins per feature, keep features with IV >= iv_min, then drop the weaker
    member of any pair whose WoE correlation exceeds corr_max."""

    def __init__(self, max_bins: int = 10, min_bin_frac: float = 0.05,
                 iv_min: float = 0.02, corr_max: float = 0.7):
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.iv_min = iv_min
        self.corr_max = corr_max

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        self.binnings_ = {
            c: fit_binning(X[c].to_numpy(), y, c, self.max_bins, self.min_bin_frac)
            for c in X.columns
        }
        ivs = pd.Series({c: b.iv for c, b in self.binnings_.items()}).sort_values(ascending=False)
        self.iv_ = ivs
        candidates = [c for c in ivs.index if ivs[c] >= self.iv_min]
        W = self._woe(X, candidates)
        corr = W.corr().abs()
        selected: list[str] = []
        for c in candidates:  # greedy, strongest IV first
            if all(corr.loc[c, s] <= self.corr_max for s in selected):
                selected.append(c)
        self.selected_ = selected
        self.feature_names_in_ = np.array(X.columns)
        return self

    def _woe(self, X: pd.DataFrame, cols) -> pd.DataFrame:
        return pd.DataFrame(
            {c: self.binnings_[c].transform(X[c].to_numpy()) for c in cols}, index=X.index
        )

    def transform(self, X):
        return self._woe(pd.DataFrame(X), self.selected_)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_)

    def binning_table(self) -> pd.DataFrame:
        return pd.concat([self.binnings_[c].table for c in self.iv_.index], ignore_index=True)
