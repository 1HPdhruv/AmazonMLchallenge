"""Per-entity score analysis + error taxonomy (exactly one class per failing entity)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ["retrieval_miss", "model_miss", "decision_miss", "no_match_false_positive",
           "multi_under_selection", "over_selection"]


def score_analysis(e, s, y, acc, entities, n_true, t=None) -> pd.DataFrame:
    d = pd.DataFrame({"e": e, "s": s, "y": y, "acc": acc})
    d = d[d.e.isin(set(entities))]
    d["rk"] = d.groupby("e")["s"].rank(ascending=False, method="first")
    g = d.groupby("e")
    out = pd.DataFrame({
        "cand_count": g.size(),
        "best_score": g["s"].max(),
        "second_score": g["s"].apply(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else np.nan),
        "best_true_score": d[d.y == 1].groupby("e")["s"].max(),
        "best_false_score": d[d.y == 0].groupby("e")["s"].max(),
        "true_rank": d[d.y == 1].groupby("e")["rk"].min(),
        "retrieved_true": g["y"].sum(),
        "n_accepted": g["acc"].sum(),
        "tp": d[d.acc & (d.y == 1)].groupby("e").size(),
    }).reindex(entities)
    out["n_true"] = n_true[entities]
    out[["cand_count", "retrieved_true", "n_accepted", "tp"]] = out[["cand_count", "retrieved_true", "n_accepted", "tp"]].fillna(0).astype(int)
    out["fp"] = out.n_accepted - out.tp
    out["fn"] = out.n_true - out.tp
    out["score_gap_top2"] = out.best_score - out.second_score
    out["true_false_gap"] = out.best_true_score - out.best_false_score
    if t is not None:
        out["n_above_t"] = d[d.s >= t].groupby("e").size().reindex(entities).fillna(0).astype(int).values
    out.index.name = "e"
    return out.reset_index()


def classify(sa: pd.DataFrame) -> pd.Series:
    def one(r):
        if r.fp == 0 and r.fn == 0:
            return "correct"
        if r.n_true == 0:
            return "no_match_false_positive"
        if r.tp > 0 and r.fp > 0:
            return "over_selection"
        if r.tp > 0 and r.fn > 0:  # fp == 0
            return "retrieval_miss" if r.retrieved_true == r.tp else "multi_under_selection"
        # tp == 0
        if r.retrieved_true == 0:
            return "retrieval_miss"
        if pd.notna(r.best_false_score) and r.best_false_score > r.best_true_score:
            return "model_miss"
        return "decision_miss"
    return sa.apply(one, axis=1)


def diagnostic_tree(sa: pd.DataFrame, recall: float, recall_target=0.95) -> list[str]:
    lines = [f"- union candidate recall on evaluated true pairs: {recall:.4f} (target {recall_target})"]
    if recall < recall_target:
        lines.append("  -> RETRIEVAL FAILURE branch: tune k / rarity weighting before anything else.")
    has = sa[(sa.retrieved_true > 0) & sa.best_false_score.notna()]
    sep = float((has.best_true_score > has.best_false_score).mean()) if len(has) else float("nan")
    lines.append(f"- entities with a retrieved true match where best-true > best-false score: {sep:.4f} (n={len(has)})")
    cls = sa["error_class"].value_counts()
    fail = cls.drop("correct", errors="ignore")
    tot = max(1, int(fail.sum()))
    for c in CLASSES:
        lines.append(f"- {c}: {int(fail.get(c, 0))} ({fail.get(c, 0) / tot:.1%} of failures)")
    return lines
