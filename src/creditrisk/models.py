"""Candidate PD models, from the regulatory-standard scorecard to gradient boosting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import MONOTONE
from .woe import WoETransformer

SEED = 42
REFERENCE_MODEL = "WoE logistic scorecard"


class WoEScorecard(ClassifierMixin, BaseEstimator):
    """WoE binning + IV/correlation filter + logistic regression, with the standard
    scorecard sign check: with WoE = ln(%good/%bad) every coefficient must be negative.
    A positive coefficient (caused by collinearity) makes points move against the bin's
    own risk ranking, so the weakest such feature is dropped and the model refitted."""

    def __init__(self, max_bins: int = 10, min_bin_frac: float = 0.05,
                 iv_min: float = 0.02, corr_max: float = 0.7, C: float = 1.0):
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.iv_min = iv_min
        self.corr_max = corr_max
        self.C = C

    def fit(self, X, y):
        self.woe_ = WoETransformer(self.max_bins, self.min_bin_frac, self.iv_min,
                                   self.corr_max).fit(X, y)
        W = self.woe_.transform(X)
        self.dropped_for_sign_ = []
        while True:
            self.lr_ = LogisticRegression(C=self.C, max_iter=2000).fit(W, y)
            coefs = pd.Series(self.lr_.coef_[0], index=W.columns)
            wrong = coefs[coefs >= 0].index
            if len(wrong) == 0 or W.shape[1] == 1:
                break
            weakest = min(wrong, key=lambda c: self.woe_.iv_[c])
            self.dropped_for_sign_.append(weakest)
            W = W.drop(columns=weakest)
        self.woe_.selected_ = list(W.columns)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.woe_.transform(X))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def build_models(feature_names: list[str]) -> dict:
    mono = [MONOTONE.get(c, 0) for c in feature_names]
    return {
        # Industry benchmark: transparent, monotone by construction, easy to validate.
        REFERENCE_MODEL: WoEScorecard(max_bins=10, min_bin_frac=0.05, iv_min=0.02, corr_max=0.7),
        "Logistic (raw features)": Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=2000)),
        ]),
        "Random forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=40, max_features="sqrt",
            n_jobs=-1, random_state=SEED),
        "Gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
        # Same learner with business-sense monotonic constraints (e.g. PD never falls
        # as arrears deepen) - the usual compromise for ML in regulated PD models.
        "Gradient boosting (monotone)": HistGradientBoostingClassifier(
            monotonic_cst=mono, random_state=SEED),
    }


GBM_SEARCH_SPACE = {
    "learning_rate": loguniform(0.02, 0.2),
    "max_iter": randint(150, 600),
    "max_leaf_nodes": randint(8, 48),
    "min_samples_leaf": randint(40, 300),
    "l2_regularization": loguniform(1e-3, 10),
    "max_features": uniform(0.5, 0.5),
}


def tune_gbm(model, X, y, n_iter: int = 20, cv=None, seed: int = SEED):
    """Random search on the TRAINING set only, scored by cross-validated AUC."""
    cv = cv or StratifiedKFold(5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        model, GBM_SEARCH_SPACE, n_iter=n_iter, scoring="roc_auc", cv=cv,
        random_state=seed, n_jobs=-1, refit=True)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def scorecard_coefficients(sc: WoEScorecard):
    """(intercept, {feature: coefficient}, all_signs_ok)."""
    coefs = dict(zip(sc.woe_.selected_, sc.lr_.coef_[0]))
    return sc.lr_.intercept_[0], coefs, all(v < 0 for v in coefs.values())


def predict_pd(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
