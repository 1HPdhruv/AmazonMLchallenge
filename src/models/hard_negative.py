"""One round of stratified hard-negative mining, TRAINING information only.

Out-of-fold scores on train_fit entities (entity-grouped K-fold) identify hard negatives in five
strata; selected negatives are up-weighted for the retrain. Mined entity IDs are asserted to be a
subset of train_fit entity IDs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.gbdt import PairModel


def oof_scores(X, y, w, e_idx, params, k=3, seed=0):
    ents = np.unique(e_idx)
    r = np.random.default_rng(seed)
    fold_of = dict(zip(ents, r.integers(0, k, len(ents))))
    f = np.array([fold_of[e] for e in e_idx])
    out = np.zeros(len(y), np.float32)
    for i in range(k):
        tr, te = f != i, f == i
        m = PairModel(params).fit(X[tr], y[tr], w[tr])
        out[te] = m.predict(X[te])
    return out


def mine(F: pd.DataFrame, y, oof, e_idx, has_true: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """Returns a weight multiplier per row and per-stratum counts."""
    neg = y == 0
    addr_sim = F["addr_tok_set"].fillna(0).values
    name_sim = F["jw"].fillna(0).values
    msi = F["max_shared_idf"].values
    shared = F["shared_idf_sum"].fillna(0).values > 0
    # data-driven "generic" cutoff: low quantile of max shared-token IDF among train rows sharing a token
    generic_cut = float(np.nanquantile(msi[shared], cfg["generic_idf_quantile"])) if shared.any() else 0.0
    strata = {
        "similar_name_diff_address": neg & (name_sim >= 0.85) & (addr_sim < 0.6),
        "similar_address_diff_name": neg & (addr_sim >= 0.8) & (name_sim < 0.7),
        "generic_shared_name": neg & shared & (np.nan_to_num(msi) <= generic_cut),
        "high_ranked_retrieval_error": neg & (F["char_rank"].fillna(99).values <= 2),
        "no_match_entity_high_score": neg & ~has_true & (oof >= cfg["no_match_min_score"]),
    }
    mult = np.ones(len(y), np.float32)
    counts = {}
    for name, m in strata.items():
        idx = np.where(m)[0]
        idx = idx[np.argsort(-oof[idx])][: cfg["per_stratum_cap"]]
        idx = idx[oof[idx] >= cfg["min_oof_score"]]
        mult[idx] = np.maximum(mult[idx], cfg["upweight"])
        counts[name] = int(len(idx))
    counts["generic_idf_cut"] = round(generic_cut, 3)
    counts["total_rows_upweighted"] = int((mult > 1).sum())
    counts["entities_touched"] = int(len(np.unique(e_idx[mult > 1])))
    return mult, counts
