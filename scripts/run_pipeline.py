"""End-to-end PD modelling pipeline.

    python scripts/run_pipeline.py              # downloads UCI data on first run
    python scripts/run_pipeline.py --fast       # quicker search / bootstrap, same steps
    python scripts/run_pipeline.py --synthetic  # offline smoke test on fake data

Outputs: reports/*.csv, reports/results.md, reports/figures/*.png and, for real data,
the results block of README.md is refreshed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from creditrisk import capital, plots, stress  # noqa: E402
from creditrisk.data import TARGET, load, make_synthetic  # noqa: E402
from creditrisk.features import build_features  # noqa: E402
from creditrisk.metrics import bootstrap_gini, psi, summarise  # noqa: E402
from creditrisk.models import (REFERENCE_MODEL, SEED, build_models, predict_pd,  # noqa: E402
                               scorecard_coefficients, tune_gbm)
from creditrisk.scorecard import grade_backtest, pd_to_score, points_table  # noqa: E402

LGD, CCF = 0.75, 0.5  # stated assumptions for the capital / stress illustration


def fmt(df: pd.DataFrame, pct=(), dec3=(), money=()) -> str:
    """Render a DataFrame as a GitHub markdown table with light formatting."""
    d = df.copy()
    for c in d.columns:
        if c in pct:
            d[c] = d[c].map(lambda v: f"{v:.2%}")
        elif c in money:
            d[c] = d[c].map(lambda v: f"{v:,.0f}")
        elif c in dec3:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--synthetic", action="store_true", help="offline run on synthetic data")
    ap.add_argument("--fast", action="store_true", help="smaller search and bootstrap")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    n_iter, n_boot = (5, 200) if args.fast else (25, 1000)

    # ------------------------------------------------------------------ data
    raw = make_synthetic() if args.synthetic else load(args.data_dir)
    X_all = build_features(raw)
    y_all = raw[TARGET].to_numpy()
    feats = list(X_all.columns)
    print(f"Data: {len(raw):,} accounts, default rate {y_all.mean():.2%}, {len(feats)} features")

    idx_tr, idx_te = train_test_split(np.arange(len(raw)), test_size=0.30,
                                      stratify=y_all, random_state=SEED)
    X_tr, X_te = X_all.iloc[idx_tr], X_all.iloc[idx_te]
    y_tr, y_te = y_all[idx_tr], y_all[idx_te]
    raw_te = raw.iloc[idx_te].reset_index(drop=True)

    # ---------------------------------------------- tuning + CV (train only)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = build_models(feats)
    tuned = {}
    for name in ["Gradient boosting", "Gradient boosting (monotone)"]:
        est, params, score = tune_gbm(models[name], X_tr, y_tr, n_iter=n_iter, cv=cv)
        models[name] = est
        tuned[name] = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                       for k, v in params.items()}
        print(f"  tuned {name}: CV AUC {score:.4f}")

    cv_rows, preds, fitted = [], {}, {}
    for name, model in models.items():
        cv_auc = cross_val_score(clone(model), X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        m = clone(model).fit(X_tr, y_tr)
        fitted[name] = m
        preds[name] = predict_pd(m, X_te)
        cv_rows.append({"model": name, "CV AUC mean": cv_auc.mean(), "CV AUC sd": cv_auc.std()})
        print(f"  {name:32s} CV AUC {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    cv_df = pd.DataFrame(cv_rows)

    # Champion chosen on TRAIN cross-validation, never on the test set.
    champion = cv_df.sort_values("CV AUC mean", ascending=False)["model"].iloc[0]

    # ------------------------------------------------- out-of-sample test
    test_df = pd.DataFrame({n: summarise(y_te, p) for n, p in preds.items()}).T
    test_df.index.name = "model"
    test_df = test_df.reset_index()
    boot = bootstrap_gini(y_te, preds, reference=REFERENCE_MODEL, n_boot=n_boot)
    results = cv_df.merge(test_df, on="model").merge(
        boot[["model", "Gini_lo", "Gini_hi", "dGini_vs_ref", "dGini_lo", "dGini_hi", "p_not_better"]],
        on="model")
    results.to_csv(out / "model_comparison.csv", index=False)

    # ------------------------------------------------------- scorecard detail
    sc = fitted[REFERENCE_MODEL]
    woe = sc.woe_
    intercept, coefs, signs_ok = scorecard_coefficients(sc)
    woe.binning_table().to_csv(out / "woe_binning.csv", index=False)
    pts = points_table(woe, intercept, coefs)
    pts.to_csv(out / "scorecard_points.csv", index=False)
    score_te = pd_to_score(preds[REFERENCE_MODEL])

    # ------------------------------------------- calibration and stability
    grades_champ = grade_backtest(y_te, preds[champion])
    grades_sc = grade_backtest(y_te, preds[REFERENCE_MODEL])
    grades_champ.to_csv(out / "grade_backtest_champion.csv", index=False)
    grades_sc.to_csv(out / "grade_backtest_scorecard.csv", index=False)
    psi_val = psi(predict_pd(fitted[champion], X_tr), preds[champion])

    # Fairness check on an attribute the model never sees.
    sex = raw_te["SEX"].map({1: "male", 2: "female"})
    fair = pd.DataFrame({"y": y_te, "pd": preds[champion], "sex": sex}).groupby("sex").agg(
        n=("y", "size"), observed_dr=("y", "mean"), mean_pd=("pd", "mean"))
    fair.to_csv(out / "fairness_by_sex.csv")

    # Permutation importance for the champion if it is a GBM, else the best GBM.
    gbm_name = champion if "boosting" in champion else max(
        ["Gradient boosting", "Gradient boosting (monotone)"],
        key=lambda n: cv_df.set_index("model").loc[n, "CV AUC mean"])
    pi = permutation_importance(fitted[gbm_name], X_te, y_te, scoring="roc_auc",
                                n_repeats=5 if args.fast else 10, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std}, index=feats)
    imp.sort_values("mean", ascending=False).to_csv(out / "permutation_importance.csv")

    # ---------------------------------------------------------- stress test
    exposure = capital.ead(raw_te["LIMIT_BAL"], raw_te["BILL_AMT1"], CCF)
    beh = stress.behavioural_stress(fitted[champion], raw_te, feats)
    sysm = stress.systematic_stress(preds[champion], capital.RHO_QRRE)
    stress_rows = {f"Behavioural: {k}": capital.portfolio_summary(v, exposure, LGD) for k, v in beh.items()}
    stress_rows.update({f"Systematic: {k}": capital.portfolio_summary(v, exposure, LGD)
                        for k, v in list(sysm.items())[1:]})
    stress_df = capital.summary_frame(stress_rows)
    stress_df.to_csv(out / "stress_test.csv")

    # ---------------------------------------------------------------- figures
    plots.roc_curves(y_te, preds, figs / "roc.png")
    calib_models = {k: preds[k] for k in dict.fromkeys([REFERENCE_MODEL, champion, "Random forest"])}
    plots.calibration(y_te, calib_models, figs / "calibration.png")
    plots.information_value(woe.iv_, woe.selected_, figs / "information_value.png")
    plots.score_distribution(score_te, y_te, figs / "score_distribution.png")
    plots.grade_backtest(grades_champ, figs / "grade_backtest.png")
    plots.feature_importance(imp, figs / "permutation_importance.png")
    short = stress_df.rename(index=lambda s: s.replace("Behavioural: ", "").replace("Systematic: ", ""))
    plots.stress_bars(short, figs / "stress.png")

    # ------------------------------------------------------------- write-up
    tbl = results[["model", "CV AUC mean", "CV AUC sd", "AUC", "Gini", "Gini_lo", "Gini_hi",
                   "KS", "Brier", "HL_pvalue"]].rename(columns={
                       "CV AUC mean": "CV AUC (train)", "CV AUC sd": "CV sd",
                       "AUC": "Test AUC", "Gini": "Test Gini", "Gini_lo": "Gini 95% lo",
                       "Gini_hi": "Gini 95% hi", "HL_pvalue": "HL p-value"})
    diff = results[results["model"] != REFERENCE_MODEL][
        ["model", "dGini_vs_ref", "dGini_lo", "dGini_hi", "p_not_better"]].rename(columns={
            "dGini_vs_ref": "ΔGini vs scorecard", "dGini_lo": "95% lo", "dGini_hi": "95% hi",
            "p_not_better": "P(Δ ≤ 0)"})
    st = stress_df[["EAD-weighted PD", "EL / EAD", "RWA density"]].reset_index().rename(
        columns={"index": "Scenario"})
    banner = ("> **SYNTHETIC DATA – smoke test only. These numbers are meaningless.**\n\n"
              if args.synthetic else "")
    dropped_txt = ("; dropped for sign: " + ", ".join(sc.dropped_for_sign_)
                   if sc.dropped_for_sign_ else "")
    signs_txt = "yes" if signs_ok else "NO - review"
    fair_txt = "; ".join(f"{k}: observed DR {r.observed_dr:.2%}, mean PD {r.mean_pd:.2%}"
                         for k, r in fair.iterrows())
    md = f"""{banner}**Data:** {len(raw):,} accounts, default rate {y_all.mean():.2%}; \
stratified 70/30 split → {len(idx_tr):,} train / {len(idx_te):,} test. \
Champion (selected on train CV AUC): **{champion}**.

**Model comparison** (CV on train; all other columns on the untouched test set; \
Gini CI from {n_boot} stratified bootstrap resamples)

{fmt(tbl, dec3=[c for c in tbl.columns if c != "model"])}

**Paired bootstrap: Gini uplift over the WoE scorecard**

{fmt(diff, dec3=[c for c in diff.columns if c != "model"])}

**Scorecard:** {len(woe.selected_)} of {len(feats)} features kept \
(IV ≥ 0.02, |ρ| ≤ 0.7, coefficient sign check{dropped_txt}); \
final signs consistent: **{signs_txt}**. Score PSI train→test (champion): **{psi_val:.4f}**.

**Master-scale back-test, champion, test set** (binomial test H0: DR = mean PD)

{fmt(grades_champ.drop(columns="defaults"), pct=["share", "mean_pd", "observed_dr"], dec3=["binom_p"])}

**Stress test, test portfolio** (LGD {LGD:.0%}, CCF {CCF:.0%}, QRRE ρ = {capital.RHO_QRRE})

{fmt(st, pct=["EAD-weighted PD", "EL / EAD", "RWA density"])}

**Protected attribute check** (SEX is excluded from the model): \
{fair_txt}.

_Run time {time.time() - t0:.0f}s._
"""
    (out / "results.md").write_text(md, encoding="utf-8")
    (out / "tuned_params.json").write_text(json.dumps(tuned, indent=2))

    readme = ROOT / "README.md"
    if not args.synthetic and readme.exists():
        txt = readme.read_text(encoding="utf-8")
        new = re.sub(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)",
                     lambda m: f"{m.group(1)}\n{md}\n{m.group(2)}", txt, flags=re.S)
        readme.write_text(new, encoding="utf-8")
        print("README results block updated.")
    print(md)
    print(f"Done in {time.time() - t0:.0f}s. Outputs in {out}")


if __name__ == "__main__":
    main()
