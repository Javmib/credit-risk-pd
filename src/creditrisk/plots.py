"""Static figures for the README (matplotlib, light theme)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

# Validated categorical order (fixed slots, never cycled) and neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2,
})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def roc_curves(y, preds: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    for (name, p), c in zip(preds.items(), SERIES):
        fpr, tpr, _ = roc_curve(y, p)
        g = 2 * np.trapezoid(tpr, fpr) - 1
        ax.plot(fpr, tpr, color=c, label=f"{name}  (Gini {g:.3f})")
    ax.set(xlabel="False positive rate (goods flagged)", ylabel="True positive rate (bads caught)",
           title="ROC on the held-out test set", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, path)


def calibration(y, preds: dict, path: Path, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(6, 5))
    hi = 0
    for (name, p), c in zip(preds.items(), SERIES):
        d = pd.DataFrame({"y": y, "p": p})
        d["b"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
        t = d.groupby("b").mean()
        hi = max(hi, t.max().max())
        ax.plot(t["p"], t["y"], marker="o", ms=5, color=c, label=name)
    ax.plot([0, hi], [0, hi], color=INK_2, lw=1, ls="--", label="Perfect calibration")
    ax.set(xlabel="Mean predicted PD (decile)", ylabel="Observed default rate",
           title="Calibration on the test set")
    ax.legend(fontsize=8.5)
    _save(fig, path)


def information_value(iv: pd.Series, selected: list[str], path: Path):
    iv = iv.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(iv) + 1.2))
    colors = [SERIES[0] if f in selected else GRID for f in iv.index]
    ax.barh(iv.index, iv.values, color=colors, height=0.7)
    ax.axvline(0.02, color=INK_2, lw=1, ls="--")
    ax.set(xlabel="Information value (train)", title="Feature strength (blue = in scorecard)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def score_distribution(score, y, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(np.percentile(score, 0.5), np.percentile(score, 99.5), 40)
    ax.hist(score[y == 0], bins=bins, density=True, alpha=0.75, color=SERIES[0], label="Non-default")
    ax.hist(score[y == 1], bins=bins, density=True, alpha=0.75, color=SERIES[1], label="Default")
    ax.set(xlabel="Scorecard points (higher = safer)", ylabel="Density",
           title="Score separation on the test set")
    ax.legend()
    _save(fig, path)


def grade_backtest(table: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(table))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, table["mean_pd"], w, color=SERIES[0], label="Mean predicted PD")
    ax.bar(x + w / 2 + 0.01, table["observed_dr"], w, color=SERIES[1], label="Observed default rate")
    for i, n in enumerate(table["n"]):
        ax.text(i, -0.045, f"n={n}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, table["grade"])
    ax.set_ylim(-0.07, max(table["observed_dr"].max(), table["mean_pd"].max()) * 1.12)
    ax.set(ylabel="Rate", title="Master-scale back-test (test set)")
    ax.legend()
    _save(fig, path)


def feature_importance(imp: pd.DataFrame, path: Path, top: int = 12):
    imp = imp.sort_values("mean").tail(top)
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(imp) + 1.2))
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color=SERIES[0], height=0.7,
            error_kw={"ecolor": INK_2, "lw": 1})
    ax.set(xlabel="Drop in test AUC when shuffled", title="Permutation importance (best GBM)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def stress_bars(table: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, col, title in zip(axes, ["EAD-weighted PD", "RWA density"],
                              ["EAD-weighted PD", "RWA density (RWA / EAD)"]):
        vals = table[col].astype(float)
        colors = [SERIES[1] if "Z =" in str(i) else SERIES[0] for i in vals.index]
        ax.bar(range(len(vals)), vals, color=colors, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(vals)), vals.index, rotation=15, fontsize=8.5)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=SERIES[0], label="Behavioural shock (re-scored)"),
                        Patch(color=SERIES[1], label="Systematic factor (Vasicek)")],
               loc="upper right", ncol=2, fontsize=8.5)
    fig.suptitle("Stress scenarios on the test portfolio", x=0.01, ha="left", fontweight="bold")
    _save(fig, path)
