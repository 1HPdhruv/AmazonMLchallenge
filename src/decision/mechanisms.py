"""All entity-level decision mechanisms behind one dispatch function.

Input: a frame with columns e (entity idx), r (record idx), s (score), sorted arbitrarily.
Output: boolean accept mask aligned with the frame.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def ranked(df):
    d = df[["e", "s"]].copy()
    d["rk"] = d.groupby("e")["s"].rank(ascending=False, method="first")
    top = d.groupby("e")["s"].transform("max")
    second = d["s"].where(d["rk"] == 2).groupby(d["e"]).transform("max").fillna(-np.inf)
    return d["rk"].values, top.values, second.values


def gmm_components(df):
    """Per-entity 2-component GMM (independent of threshold -> computed once, then cached).
    Returns (in_high_component, high_component_mean) per row; entities with <2 distinct scores
    get in_high=True and mean=score (degenerate -> behaves like a plain threshold)."""
    in_hi = np.ones(len(df), bool)
    hi_mean = df["s"].values.astype(float).copy()
    s = df["s"].values
    for e, idx in df.groupby("e").indices.items():
        x = s[idx]
        if len(x) < 2 or np.ptp(x) < 1e-6:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = GaussianMixture(2, random_state=0).fit(x.reshape(-1, 1))
        hi = int(np.argmax(g.means_.ravel()))
        in_hi[idx] = g.predict(x.reshape(-1, 1)) == hi
        hi_mean[idx] = g.means_.ravel()[hi]
    return in_hi, hi_mean


def decide(df: pd.DataFrame, mech: str, p: dict, rank_cache=None) -> np.ndarray:
    s = df["s"].values
    if mech == "threshold":
        return s >= p["t"]
    rk, top, second = rank_cache if rank_cache is not None else ranked(df)
    if mech == "top1":
        return rk == 1
    if mech == "top1_t":
        return (rk == 1) & (s >= p["t"])
    if mech == "top1_margin":
        return (rk == 1) & (s >= p["t"]) & ((top - second) >= p["m"])
    if mech == "multi":  # top-1 clears t; others accepted if >= t and within delta of top
        return (top >= p["t"]) & (s >= p["t"]) & ((top - s) <= p["delta"])
    if mech == "multi_t2":  # top-1 clears t; additional candidates need only the lower floor t2
        return (top >= p["t"]) & ((rk == 1) | (s >= p["t2"])) & (s >= min(p["t"], p["t2"]))
    if mech == "abs2":  # top-1 needs t1; every other candidate needs its own ABSOLUTE bar t2 (may exceed t1)
        return ((rk == 1) & (s >= p["t1"])) | ((rk > 1) & (top >= p["t1"]) & (s >= p["t2"]))
    if mech == "gmm":
        in_hi, hi_mean = p["cache"] if "cache" in p else gmm_components(df)
        return in_hi & (s >= p["t"]) & (hi_mean >= p["t"])
    raise ValueError(mech)


def record_exclusive(df: pd.DataFrame, acc: np.ndarray, mode: str = "accepted") -> np.ndarray:
    """A record belongs to at most one Source-1 entity (Source 1 is the deduplicated reference source).
    mode='accepted': among ACCEPTED pairs, keep a record only for its highest-scoring entity (ties kept).
    mode='all'     : keep an accepted (e, r) only if e is r's strict best over ALL candidate entities,
                     accepted or not; exact ties drop the record for every tied entity."""
    if mode == "all":
        best = df.groupby("r")["s"].transform("max").values
        n_best = (df["s"].values >= best - 1e-12).astype(int)
        n_at_best = pd.Series(n_best).groupby(df["r"].values).transform("sum").values
        return acc & (df["s"].values >= best - 1e-12) & (n_at_best == 1)
    d = df.loc[acc, ["r", "s"]]
    if d.empty:
        return acc
    best = d.groupby("r")["s"].transform("max")
    keep = np.zeros(len(df), bool)
    keep[np.where(acc)[0][(d["s"].values >= best.values)]] = True
    return keep


_T = np.round(np.concatenate([np.arange(0.05, 0.95, 0.025), [0.95, 0.965, 0.98, 0.99, 0.995]]), 3)
_T_COARSE = np.round(np.concatenate([np.arange(0.05, 0.95, 0.05), [0.95, 0.98, 0.99]]), 3)
GRIDS = {
    "threshold": [{"t": t} for t in _T],
    "top1": [{}],
    "top1_t": [{"t": t} for t in _T],
    "top1_margin": [{"t": t, "m": m} for t in _T_COARSE
                    for m in (0.0, 0.05, 0.1, 0.2, 0.3)],
    "multi": [{"t": t, "delta": d} for t in _T
              for d in (0.05, 0.10, 0.15, 0.20)],
    "multi_t2": [{"t": t, "t2": t2} for t in _T_COARSE for t2 in _T_COARSE if t2 <= t],
    "abs2": [{"t1": t1, "t2": t2} for t1 in _T_COARSE for t2 in _T_COARSE],
    "gmm": [{"t": t} for t in np.round(np.arange(0.1, 0.91, 0.1), 3)] + [{"t": 0.95}, {"t": 0.99}],
}
