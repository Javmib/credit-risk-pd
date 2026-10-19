**Data:** 30,000 accounts, default rate 22.12%; stratified 70/30 split → 21,000 train / 9,000 test. Champion (selected on train CV AUC): **Gradient boosting**.

**Model comparison** (CV on train; all other columns on the untouched test set; Gini CI from 1000 stratified bootstrap resamples)

| model | CV AUC (train) | CV sd | Test AUC | Test Gini | Gini 95% lo | Gini 95% hi | KS | Brier | HL p-value |
|---|---|---|---|---|---|---|---|---|---|
| WoE logistic scorecard | 0.757 | 0.009 | 0.755 | 0.509 | 0.483 | 0.536 | 0.406 | 0.140 | 0.000 |
| Logistic (raw features) | 0.761 | 0.009 | 0.746 | 0.492 | 0.466 | 0.518 | 0.394 | 0.143 | 0.000 |
| Random forest | 0.788 | 0.008 | 0.777 | 0.555 | 0.531 | 0.579 | 0.421 | 0.135 | 0.201 |
| Gradient boosting | 0.790 | 0.007 | 0.779 | 0.559 | 0.535 | 0.583 | 0.428 | 0.135 | 0.897 |
| Gradient boosting (monotone) | 0.786 | 0.010 | 0.777 | 0.553 | 0.530 | 0.578 | 0.428 | 0.135 | 0.103 |

**Paired bootstrap: Gini uplift over the WoE scorecard**

| model | ΔGini vs scorecard | 95% lo | 95% hi | P(Δ ≤ 0) |
|---|---|---|---|---|
| Logistic (raw features) | -0.017 | -0.029 | -0.006 | 0.999 |
| Random forest | 0.046 | 0.031 | 0.060 | 0.000 |
| Gradient boosting | 0.050 | 0.035 | 0.064 | 0.000 |
| Gradient boosting (monotone) | 0.044 | 0.033 | 0.056 | 0.000 |

**Scorecard:** 9 of 19 features kept (IV ≥ 0.02, |ρ| ≤ 0.7, coefficient sign check; dropped for sign: n_paid_in_full_6m); final signs consistent: **yes**. Score PSI train→test (champion): **0.0005**.

**Master-scale back-test, champion, test set** (binomial test H0: DR = mean PD)

| grade | n | share | mean_pd | observed_dr | binom_p |
|---|---|---|---|---|---|
| G2 | 643 | 7.14% | 4.22% | 3.89% | 0.768 |
| G3 | 2167 | 24.08% | 7.32% | 7.84% | 0.343 |
| G4 | 1755 | 19.50% | 12.48% | 13.56% | 0.170 |
| G5 | 2047 | 22.74% | 18.89% | 18.08% | 0.366 |
| G6 | 986 | 10.96% | 32.00% | 33.67% | 0.260 |
| G7 | 538 | 5.98% | 48.75% | 47.03% | 0.438 |
| G8 | 864 | 9.60% | 70.91% | 69.79% | 0.477 |

**Stress test, test portfolio** (LGD 75%, CCF 50%, QRRE ρ = 0.04)

| Scenario | EAD-weighted PD | EL / EAD | RWA density |
|---|---|---|---|
| Behavioural: Baseline | 18.14% | 13.61% | 148.37% |
| Behavioural: Adverse | 19.04% | 14.28% | 155.88% |
| Behavioural: Severely adverse | 22.22% | 16.67% | 159.27% |
| Systematic: 1-in-10 (Z = -1.28) | 24.26% | 18.20% | 171.87% |
| Systematic: 1-in-25 (Z = -1.75) | 26.80% | 20.10% | 179.28% |

**Protected attribute check** (SEX is excluded from the model): female: observed DR 20.99%, mean PD 21.59%; male: observed DR 23.86%, mean PD 22.69%.

_Run time 186s._
