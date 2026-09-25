"""Entity-level macro F0.5 (primary metric) + stratified breakdowns.

Per Source-1 entity with truth set T and prediction P:
  T = {} and P = {}      -> 1.0
  T = {} and P != {}     -> 0.0   (no partial credit for no-match false positives)
  T != {} and P = {}     -> 0.0
  otherwise              -> F-beta(precision=|P∩T|/|P|, recall=|P∩T|/|T|), beta = 0.5
Macro = unweighted mean over entities.
NOTE: this is our implementation of the metric described in the directive; the official scorer,
when available, is authoritative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA = 0.5


def fbeta(p: set, t: set, beta=BETA) -> float:
    if not t and not p:
        return 1.0
    if not t or not p:
        return 0.0
    tp = len(p & t)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(p), tp / len(t)
    b2 = beta * beta
    return (1 + b2) * pr * rc / (b2 * pr + rc)


def per_entity(e_idx, acc, y, entities, n_true, cand_count) -> pd.DataFrame:
    """Vectorized per-entity scoring.
    e_idx/acc/y: candidate-level entity index, accept mask, label (1 = true pair).
    entities: entity indices to evaluate; n_true/cand_count: arrays indexed by entity index
    (n_true counts ALL true matches, including ones retrieval never produced)."""
    n = len(n_true)
    tp = np.bincount(e_idx, weights=(acc & (y == 1)).astype(float), minlength=n)[entities]
    npred = np.bincount(e_idx, weights=acc.astype(float), minlength=n)[entities]
    nt = n_true[entities].astype(float)
    b2 = BETA * BETA
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(npred > 0, tp / npred, 0.0)
        rc = np.where(nt > 0, tp / nt, 0.0)
        f = np.where(tp > 0, (1 + b2) * pr * rc / (b2 * pr + rc), 0.0)
    f = np.where((nt == 0) & (npred == 0), 1.0, f)
    df = pd.DataFrame({"e": entities, "f05": f, "n_true": nt.astype(int), "n_pred": npred.astype(int),
                       "tp": tp.astype(int), "fp": (npred - tp).astype(int), "fn": (nt - tp).astype(int),
                       "cand_count": cand_count[entities]})
    df["gt_type"] = np.where(df.n_true == 0, "no_match", np.where(df.n_true == 1, "single", "multi"))
    df["cand_bucket"] = pd.cut(df.cand_count, [-1, 1, 4, 20, 10 ** 9], labels=["0-1", "2-4", "5-20", "20+"]).astype(str)
    return df


def macro_f05(e_idx, acc, y, entities, n_true) -> float:
    n = len(n_true)
    tp = np.bincount(e_idx, weights=(acc & (y == 1)).astype(float), minlength=n)[entities]
    npred = np.bincount(e_idx, weights=acc.astype(float), minlength=n)[entities]
    nt = n_true[entities].astype(float)
    b2 = BETA * BETA
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tp > 0, (1 + b2) * tp / (b2 * nt + npred), 0.0)  # algebraic form of F-beta
    f = np.where((nt == 0) & (npred == 0), 1.0, f)
    return float(f.mean())


def summarize(df: pd.DataFrame) -> dict:
    nm = df[df.gt_type == "no_match"]
    mm = df[df.gt_type == "multi"]
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    return {
        "n": int(len(df)),
        "macro_f05": round(float(df.f05.mean()), 4) if len(df) else float("nan"),
        "pair_precision": round(float(tp / max(1, tp + fp)), 4),
        "pair_recall": round(float(tp / max(1, tp + fn)), 4),
        "no_match_fp_rate": round(float((nm.n_pred > 0).mean()), 4) if len(nm) else None,
        "multi_precision": round(float(mm.tp.sum() / max(1, mm.tp.sum() + mm.fp.sum())), 4) if len(mm) else None,
        "multi_recall": round(float(mm.tp.sum() / max(1, mm.tp.sum() + mm.fn.sum())), 4) if len(mm) else None,
        "fp_per_entity": round(float(df.fp.mean()), 4),
        "fn_per_entity": round(float(df.fn.mean()), 4),
        "accepted_per_entity": round(float(df.n_pred.mean()), 4),
    }


def stratified(df: pd.DataFrame, extra_cols=()) -> dict:
    out = {"overall": summarize(df)}
    for col in ("cand_bucket", "gt_type", *extra_cols):
        if col in df:
            for k, g in df.groupby(col):
                out[f"{col}={k}"] = summarize(g)
    return out


def pred_sets(cand_e, cand_r, accept, entity_ids) -> dict:
    """entity index -> frozenset(record idx) of accepted candidates (entities with none -> empty)."""
    out = {e: frozenset() for e in entity_ids}
    d = pd.DataFrame({"e": cand_e[accept], "r": cand_r[accept]})
    for e, g in d.groupby("e")["r"]:
        if e in out:
            out[e] = frozenset(g.values)
    return out


def paired_bootstrap(f_a: np.ndarray, f_b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over entities for mean(f_b - f_a). f_a/f_b are per-entity F0.5 aligned by entity."""
    d = np.asarray(f_b, float) - np.asarray(f_a, float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(d), (n_boot, len(d)))
    bs = d[idx].mean(axis=1)
    return {"diff": round(float(d.mean()), 4), "ci90": [round(float(np.quantile(bs, 0.05)), 4),
                                                         round(float(np.quantile(bs, 0.95)), 4)],
            "p_improve": round(float((bs > 0).mean()), 3)}
