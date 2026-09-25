"""GBDT pair classifier (XGBoost). Binary objective by default; ranking as a comparison point."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def entity_weights(e_idx: np.ndarray, scheme: str) -> np.ndarray:
    """'none' -> 1; 'inv_count' -> 1 / candidate_count(entity) (equal gradient mass per entity)."""
    if scheme == "none":
        return np.ones(len(e_idx), np.float32)
    if scheme == "inv_count":
        cnt = pd.Series(e_idx).map(pd.Series(e_idx).value_counts()).values
        w = 1.0 / cnt
        return (w * len(w) / w.sum()).astype(np.float32)  # rescale so mean weight = 1
    raise ValueError(scheme)


class PairModel:
    def __init__(self, params: dict, objective: str = "binary"):
        self.params = dict(params)
        self.objective = objective
        self.booster = None
        self.features = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray | None = None, groups: np.ndarray | None = None):
        self.features = list(X.columns)
        p = dict(self.params)
        n_rounds = p.pop("n_estimators")
        if self.objective == "binary":
            p["objective"] = "binary:logistic"
            p["eval_metric"] = "logloss"
            p["scale_pos_weight"] = float((y == 0).sum() / max(1, (y == 1).sum())) \
                if p.pop("auto_scale_pos_weight", True) else 1.0
            d = xgb.DMatrix(X.values, label=y, weight=w, feature_names=self.features, missing=np.nan)
        else:  # ranking comparison point: groups must be sorted by entity
            p.pop("auto_scale_pos_weight", None)
            p["objective"] = "rank:pairwise"
            order = np.argsort(groups, kind="stable")
            d = xgb.DMatrix(X.values[order], label=y[order], feature_names=self.features, missing=np.nan)
            d.set_group(np.bincount(pd.factorize(groups[order])[0]))
        self.booster = xgb.train(p, d, num_boost_round=n_rounds)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(X[self.features].values, feature_names=self.features, missing=np.nan)
        return self.booster.predict(d)

    def importance(self, top=25):
        g = self.booster.get_score(importance_type="gain")
        return dict(sorted(g.items(), key=lambda x: -x[1])[:top])

    def save(self, path):
        self.booster.save_model(path)
