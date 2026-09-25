"""Decision-layer oracle (Section 9): frozen candidates + scores + truth; vary only the mechanism.

Protocol (no circularity):
  * each mechanism's parameters are tuned on the CALIB fold (carved from train)
  * mechanisms are compared on VAL
  * isotonic calibration is fit on CALIB, so every variant is out-of-sample on VAL
Decisions are computed on the full candidate frame (all entities at once, as at inference time),
then scored on the entity subset in question; this matters only for record_exclusive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision.mechanisms import GRIDS, decide, gmm_components, ranked, record_exclusive
from src.evaluation.entity_metrics import macro_f05, paired_bootstrap, per_entity, stratified


class Frame:
    def __init__(self, e, r, s, y):
        self.df = pd.DataFrame({"e": e, "r": r, "s": s})
        self.y = y
        self.ranked = ranked(self.df)
        self._gmm = None

    def gmm(self):
        if self._gmm is None:
            self._gmm = gmm_components(self.df)
        return self._gmm

    def accept(self, mech, p, excl=False):
        """excl: False/'off' | True/'accepted' (arbitrate among accepted pairs) | 'all' (a record may only
        go to its best-scoring entity over ALL candidates; exact ties are dropped)."""
        pp = dict(p)
        if mech == "gmm":
            pp["cache"] = self.gmm()
        acc = decide(self.df, mech, pp, self.ranked)
        mode = {False: "off", True: "accepted", None: "off"}.get(excl, excl)
        if mode == "off":
            return acc
        return record_exclusive(self.df, acc, mode)


def _param_key(p: dict):
    """Precision-leaning order: larger thresholds first, smaller deltas first."""
    order = ["t", "t1", "t2", "m", "delta"]
    return tuple((-float(p[k]) if k != "delta" else float(p[k])) for k in order if k in p)


def tune(fr: Frame, mech, ents, n_true, excl=False, tie_break="low"):
    """Grid search on calib. tie_break='low' keeps the first (most permissive) grid point on a plateau
    of equal calib F0.5; 'high' prefers the most conservative parameters among exact ties."""
    scored = [(macro_f05(fr.df["e"].values, fr.accept(mech, p, excl), fr.y, ents, n_true), i, p)
              for i, p in enumerate(GRIDS[mech])]
    best = max(f for f, _, _ in scored)
    ties = [(i, p) for f, i, p in scored if f >= best - 1e-9]
    if tie_break == "high":
        best_p = sorted(ties, key=lambda ip: _param_key(ip[1]))[0][1]
    else:
        best_p = ties[0][1]
    return best_p, best


def run_oracle(frames: dict, calib_ents, val_ents, n_true, cand_count, ent_attrs: pd.DataFrame,
               mechanisms=("threshold", "top1", "top1_t", "top1_margin", "multi", "multi_t2", "gmm"),
               with_exclusive=True, collapse_tol=0.05, default="cal:multi", p_keep=0.9):
    """frames: {'raw': Frame, 'cal': Frame}. Returns (results table, detail dict)."""
    rows, detail, vecs = [], {}, {}
    for variant, fr in frames.items():
        for mech in mechanisms:
            for excl in ([False, "all"] if with_exclusive else [False]):   # '+excl' = all-candidate exclusivity
                p, f_cal = tune(fr, mech, calib_ents, n_true, excl)
                acc = fr.accept(mech, p, excl)
                pe = per_entity(fr.df["e"].values, acc, fr.y, val_ents, n_true, cand_count)
                pe = pe.merge(ent_attrs, on="e", how="left")
                st = stratified(pe, extra_cols=("country", "true_sources"))
                name = f"{variant}:{mech}{'+excl' if excl else ''}"
                vecs[name] = pe["f05"].values
                detail[name] = {"params": p, "calib_f05": round(f_cal, 4), "val": st}
                rows.append({"mechanism": name, "params": {k: v for k, v in p.items()},
                             "calib_f05": round(f_cal, 4), **{f"val_{k}": v for k, v in st["overall"].items()},
                             **{f"f05[{k}]": v["macro_f05"] for k, v in st.items() if k != "overall"}})
    # hybrid by candidate-count bucket: per bucket choose the best mechanism ON CALIB, score on VAL
    for variant, fr in frames.items():
        pe_cal = per_entity(fr.df["e"].values, np.zeros(len(fr.df), bool), fr.y, calib_ents, n_true, cand_count)
        bucket_of = dict(zip(pe_cal.e, pe_cal.cand_bucket))
        pe_val = per_entity(fr.df["e"].values, np.zeros(len(fr.df), bool), fr.y, val_ents, n_true, cand_count)
        bucket_of.update(dict(zip(pe_val.e, pe_val.cand_bucket)))
        row_bucket = pd.Series(fr.df["e"].values).map(bucket_of).values
        acc = np.zeros(len(fr.df), bool)
        choice = {}
        for b in ("0-1", "2-4", "5-20", "20+"):
            ents_b = np.array([e for e in calib_ents if bucket_of.get(e) == b])
            if len(ents_b) == 0:
                continue
            best = (-1, None, None)
            for mech in ("top1_t", "top1_margin", "multi", "threshold"):
                p, f = tune(fr, mech, ents_b, n_true)
                if f > best[0] + 1e-9:
                    best = (f, mech, p)
            choice[b] = {"mechanism": best[1], "params": best[2], "calib_f05": round(best[0], 4)}
            acc |= fr.accept(best[1], best[2]) & (row_bucket == b)
        pe = per_entity(fr.df["e"].values, acc, fr.y, val_ents, n_true, cand_count).merge(ent_attrs, on="e", how="left")
        st = stratified(pe, extra_cols=("country", "true_sources"))
        name = f"{variant}:hybrid_by_bucket"
        vecs[name] = pe["f05"].values
        detail[name] = {"params": choice, "val": st}
        rows.append({"mechanism": name, "params": choice, "calib_f05": None,
                     **{f"val_{k}": v for k, v in st["overall"].items()},
                     **{f"f05[{k}]": v["macro_f05"] for k, v in st.items() if k != "overall"}})
    tab = pd.DataFrame(rows).sort_values("val_macro_f05", ascending=False).reset_index(drop=True)
    # selection rule: best overall that does not collapse in the low-count or no-match strata
    guard_cols = [c for c in ("f05[cand_bucket=2-4]", "f05[gt_type=no_match]", "f05[cand_bucket=0-1]") if c in tab]
    best_in = {c: tab[c].max() for c in guard_cols}
    tab["collapses"] = [
        [c for c in guard_cols if pd.notna(row[c]) and row[c] < best_in[c] - collapse_tol] for _, row in tab.iterrows()]
    tab["bootstrap_vs_default"] = [paired_bootstrap(vecs[default], vecs[m]) for m in tab["mechanism"]]
    eligible = tab[tab["collapses"].map(len) == 0]
    top = (eligible if len(eligible) else tab).iloc[0]["mechanism"]
    # significance gate: a non-default mechanism must beat the default with P(improve) >= p_keep
    bt = paired_bootstrap(vecs[default], vecs[top])
    winner = top if (top == default or bt["p_improve"] >= p_keep) else default
    detail["_selection"] = {"top_by_val": top, "default": default, "bootstrap_top_vs_default": bt, "winner": winner}
    return tab, detail, winner


def low_count_simulation(fr: Frame, calib_ents, val_ents, n_true, cand_count,
                         ns=(1, 2, 3, 4), mechs=("top1_t", "top1_margin", "multi", "gmm")):
    """Controlled test of mechanisms at small candidate counts: every entity's candidate list is
    truncated to its top-n by score (simulating a much tighter retriever), then each mechanism is
    tuned on calib and scored on val. Needed because the synthetic retriever never produces <5
    candidates, so the natural 1 / 2-4 strata are empty."""
    rows = []
    d = fr.df.assign(y=fr.y)
    d["rk"] = d.groupby("e")["s"].rank(ascending=False, method="first")
    for n in ns:
        sub = d[d.rk <= n].reset_index(drop=True)
        f = Frame(sub["e"].values, sub["r"].values, sub["s"].values, sub["y"].values.astype(np.int8))
        for mech in mechs:
            p, f_cal = tune(f, mech, calib_ents, n_true)
            acc = f.accept(mech, p)
            pe = per_entity(sub["e"].values, acc, f.y, val_ents, n_true, cand_count)
            st = stratified(pe)
            rows.append({"n_candidates": n, "mechanism": mech, "params": p, "calib_f05": round(f_cal, 4),
                         "val_macro_f05": st["overall"]["macro_f05"],
                         "val_no_match_fp": st["overall"]["no_match_fp_rate"],
                         "val_f05_no_match": st.get("gt_type=no_match", {}).get("macro_f05"),
                         "val_f05_single": st.get("gt_type=single", {}).get("macro_f05")})
    return pd.DataFrame(rows)
