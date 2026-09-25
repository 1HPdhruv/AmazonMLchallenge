"""Pair feature generation.

Groups (each can be toggled in configs/model.yaml -> features):
  lexical, idf, fs (Fellegi-Sunter-style heuristic weight; EXPERIMENTAL), address,
  missingness, retrieval, relative_entity (candidate-relative, entity side), relative_record
  (candidate-relative, record side).
All learned statistics (token IDF, FS m/u) come from objects fit on train_fit only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from src.preprocessing.tokenize import ok
from src.retrieval.union import active_channels


def _rowdot(A, B, ei, rj, chunk=50000):
    out = np.empty(len(ei), np.float32)
    for s in range(0, len(ei), chunk):
        a, b = A[ei[s:s + chunk]], B[rj[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _eq(a, b):
    """1/0 when both present, NaN otherwise."""
    return np.array([np.nan if not (ok(x) and ok(y)) else float(x == y)
                     for x, y in zip(a, b)], np.float32)


def _jw(a, b):
    return np.array([np.nan if not (ok(x) and ok(y)) else JaroWinkler.normalized_similarity(x, y)
                     for x, y in zip(a, b)], np.float32)


class TokenIdf:
    """word -> IDF, backed by the train-fit word TF-IDF model."""

    def __init__(self, word_model, rare_quantile: float, train_docs):
        self.m = word_model
        self.cache = {}
        vals = np.concatenate([self.get(d.split()) for d in train_docs if d] or [np.zeros(1)])
        self.rare_thr = float(np.quantile(vals, rare_quantile)) if len(vals) else 0.0

    def get(self, toks):
        miss = sorted(t for t in set(toks) if t not in self.cache)
        if miss:
            X = self.m.hv.transform(miss)
            for i, t in enumerate(miss):
                idx = X[i].indices
                self.cache[t] = float(self.m.idf[idx[0]]) if len(idx) else 0.0
        return np.array([self.cache[t] for t in toks], float)


def idf_features(ea, rb, tidf: TokenIdf):
    n = len(ea)
    f = {k: np.zeros(n, np.float32) for k in
         ("shared_idf_sum", "max_shared_idf", "unshared_idf_sum", "rare_shared", "rare_unshared_ent",
          "weighted_jaccard", "unseen_shared", "unseen_unshared")}
    for i, (x, y) in enumerate(zip(ea, rb)):
        A, B = set(x.split()), set(y.split())
        if not A or not B:
            for k in f:
                f[k][i] = np.nan
            continue
        sh, un = sorted(A & B), sorted(A ^ B)
        si, ui = tidf.get(sh), tidf.get(un)
        f["shared_idf_sum"][i] = si.sum()
        f["max_shared_idf"][i] = si.max() if len(si) else 0.0
        f["unshared_idf_sum"][i] = ui.sum()
        f["rare_shared"][i] = (si >= tidf.rare_thr).sum()
        f["rare_unshared_ent"][i] = (tidf.get(sorted(A - B)) >= tidf.rare_thr).sum()
        f["unseen_shared"][i] = tidf.m.n_unseen(sh)       # tokens never seen in train: own state,
        f["unseen_unshared"][i] = tidf.m.n_unseen(un)     # not "rarest" (France zero-shot)
        tot = si.sum() + ui.sum()
        f["weighted_jaccard"][i] = si.sum() / tot if tot > 0 else 0.0
    return f


class FellegiSunter:
    """Heuristic m/u agreement weights, fit on train_fit labelled candidate pairs only.
    Conditional independence is certainly violated; this is a GBDT input, not a calibrated LLR."""
    FIELDS = ["name_key_eq", "jw_hi", "house_eq", "postal_eq", "city_eq", "street_hi"]

    def fit(self, agree: pd.DataFrame, y: np.ndarray):
        self.w = {}
        for f in self.FIELDS:
            v = agree[f].values
            ok = ~np.isnan(v)
            m = (np.sum((v == 1) & ok & (y == 1)) + 1) / (np.sum(ok & (y == 1)) + 2)
            u = (np.sum((v == 1) & ok & (y == 0)) + 1) / (np.sum(ok & (y == 0)) + 2)
            self.w[f] = (float(np.log2(m / u)), float(np.log2((1 - m) / (1 - u))))
        return self

    def score(self, agree: pd.DataFrame):
        s = np.zeros(len(agree))
        for f, (wa, wd) in self.w.items():
            v = agree[f].values
            s += np.where(v == 1, wa, np.where(v == 0, wd, 0.0))
        return s.astype(np.float32)


def fs_agreements(F: pd.DataFrame) -> pd.DataFrame:
    def thr(col, t):
        v = F[col].values
        return np.where(np.isnan(v), np.nan, (v >= t).astype(float))
    return pd.DataFrame({"name_key_eq": F["name_key_eq"].values, "jw_hi": thr("jw", 0.92),
                         "house_eq": F["house_eq"].values, "postal_eq": F["postal_eq"].values,
                         "city_eq": F["city_eq"].values, "street_hi": thr("street_jw", 0.9)})


def _group_rank(df, key, col, ascending=False):
    g = df.groupby(key)[col]
    rank = g.rank(ascending=ascending, method="min")
    gap = g.transform("max") - df[col] if not ascending else df[col] - g.transform("min")
    pct = g.rank(pct=True, ascending=True)
    return rank.astype(np.float32), gap.astype(np.float32), pct.astype(np.float32)


def build_features(cand: pd.DataFrame, E: pd.DataFrame, R: pd.DataFrame, e_enc, r_enc,
                   tidf: TokenIdf, fs: FellegiSunter | None, groups: dict, rec_source_codes: np.ndarray):
    """Uses only competition fields: business_name, business_address (and components parsed from
    it), country, and the record's source (known from its file / id prefix)."""
    ei, rj = cand["e"].values, cand["r"].values
    e = E.iloc[ei].reset_index(drop=True)
    r = R.iloc[rj].reset_index(drop=True)
    F = {}
    en, rn = e["name_clean"].values, r["name_clean"].values
    # always computed (needed by FS and error analysis); dropped later if group disabled
    F["jw"] = _jw(en, rn)
    F["name_key_eq"] = _eq(e["name_key"].values, r["name_key"].values)
    F["house_eq"] = _eq(e["house"].values, r["house"].values)
    F["postal_eq"] = _eq(e["postal_n"].values, r["postal_n"].values)
    F["city_eq"] = _eq(e["city_n"].values, r["city_n"].values)
    F["street_jw"] = _jw(e["street_core"].values, r["street_core"].values)
    F["char_cos"] = _rowdot(e_enc["char"], r_enc["char"], ei, rj)
    if "addrtxt" in e_enc:   # parse-free, IDF-weighted address similarity (train-fit IDF)
        F["addr_text_cos"] = _rowdot(e_enc["addrtxt"], r_enc["addrtxt"], ei, rj)
    F["word_cos"] = _rowdot(e_enc["word"], r_enc["word"], ei, rj)
    F["addr_tok_set"] = np.array([np.nan if not x.strip() or not y.strip() else fuzz.token_set_ratio(x, y) / 100
                                  for x, y in zip(e["addr_fold"].values, r["addr_fold"].values)], np.float32)
    if groups.get("lexical", True):
        F["lev"] = np.array([fuzz.ratio(x, y) / 100 for x, y in zip(en, rn)], np.float32)
        F["tok_sort"] = np.array([fuzz.token_sort_ratio(x, y) / 100 for x, y in zip(en, rn)], np.float32)
        F["tok_set"] = np.array([fuzz.token_set_ratio(x, y) / 100 for x, y in zip(en, rn)], np.float32)
        F["partial"] = np.array([fuzz.partial_ratio(x, y) / 100 for x, y in zip(en, rn)], np.float32)
        F["lev_fold_full"] = np.array([fuzz.ratio(x, y) / 100 for x, y in
                                       zip(e["name_fold"].values, r["name_fold"].values)], np.float32)
        ja, co, lr, ld, ft = [], [], [], [], []
        for x, y in zip(en, rn):
            A, B = set(x.split()), set(y.split())
            if not A or not B:
                ja.append(np.nan), co.append(np.nan), lr.append(np.nan), ld.append(np.nan), ft.append(np.nan)
                continue
            ja.append(len(A & B) / len(A | B))
            co.append(len(A & B) / min(len(A), len(B)))
            lr.append(min(len(x), len(y)) / max(len(x), len(y)))
            ld.append(abs(len(x) - len(y)))
            ft.append(float(x.split()[0] == y.split()[0]))
        F["jaccard"], F["containment"], F["len_ratio"], F["len_diff"], F["first_tok_eq"] = \
            (np.array(v, np.float32) for v in (ja, co, lr, ld, ft))
        F["legal_eq"] = _eq(e["legal"].values, r["legal"].values)
    if groups.get("phonetic", False):  # transliteration / typo evidence (Metaphone + NYSIIS token codes)
        pm, pn, pk = [], [], []
        for a1, b1, a2, b2 in zip(e["name_mp"].values, r["name_mp"].values, e["name_ny"].values, r["name_ny"].values):
            A, B = set(a1.split()), set(b1.split())
            C, D = set(a2.split()), set(b2.split())
            pm.append(len(A & B) / len(A | B) if A and B else np.nan)
            pn.append(len(C & D) / len(C | D) if C and D else np.nan)
            pk.append(float(A == B) if A and B else np.nan)
        F["phon_mp_jacc"], F["phon_ny_jacc"], F["phon_key_eq"] = (np.array(v, np.float32) for v in (pm, pn, pk))
    if groups.get("idf", True):
        F.update(idf_features(en, rn, tidf))
    if groups.get("address", True):
        F["unit_eq"] = _eq(e["unit"].values, r["unit"].values)
        F["region_eq"] = _eq(e["region_n"].values, r["region_n"].values)
        F["country_eq"] = _eq(e["country_n"].values, r["country_n"].values)
        F["city_jw"] = _jw(e["city_n"].values, r["city_n"].values)
        F["street_tok_set"] = np.array([np.nan if not (ok(x) and ok(y)) else fuzz.token_set_ratio(x, y) / 100
                                        for x, y in zip(e["street_norm"].values, r["street_norm"].values)], np.float32)
        F["postal_prefix_eq"] = _eq(e["postal_n"].map(lambda p: p[:3] if ok(p) else None).values,
                                    r["postal_n"].map(lambda p: p[:3] if ok(p) else None).values)
        no, cont = [], []
        for a, b, x, y in zip(e["numbers"].values, r["numbers"].values, e["addr_fold"].values, r["addr_fold"].values):
            A, B = set(a), set(b)
            no.append(len(A & B) / len(A | B) if (A and B) else np.nan)
            X, Y = set(x.split()), set(y.split())
            cont.append(len(X & Y) / len(X) if (X and Y) else np.nan)
        F["num_overlap"], F["addr_containment"] = np.array(no, np.float32), np.array(cont, np.float32)
    if groups.get("missingness", True):
        F["e_has_name"] = e["has_name"].values.astype(np.float32)
        F["r_has_name"] = r["has_name"].values.astype(np.float32)
        F["e_has_addr"] = e["has_addr"].values.astype(np.float32)
        F["r_has_addr"] = r["has_addr"].values.astype(np.float32)
        F["r_has_postal"] = r["postal_n"].map(ok).values.astype(np.float32)
        F["r_has_city"] = r["city_n"].map(ok).values.astype(np.float32)
        F["r_source"] = rec_source_codes[rj].astype(np.float32)
    if groups.get("retrieval", True):
        for ch in active_channels(cand):
            F[f"{ch}_score"] = cand[f"{ch}_score"].values.astype(np.float32)
            F[f"{ch}_rank"] = cand[f"{ch}_rank"].values.astype(np.float32)
        F["n_channels"] = cand["n_channels"].values.astype(np.float32)
    F = pd.DataFrame(F)
    if groups.get("fs", True) and fs is not None:
        F["fs_weight"] = fs.score(fs_agreements(F))
    tmp = pd.DataFrame({"e": ei, "r": rj, "char_cos": F["char_cos"], "jw": F["jw"].fillna(0),
                        "addr": F["addr_tok_set"].fillna(0)})
    if groups.get("relative_entity", False):
        F["cand_count"] = tmp.groupby("e")["e"].transform("size").astype(np.float32).values
        for col in ("char_cos", "jw", "addr"):
            rk, gap, pct = _group_rank(tmp, "e", col)
            F[f"rel_{col}_rank"], F[f"rel_{col}_gap"], F[f"rel_{col}_pct"] = rk.values, gap.values, pct.values
    if groups.get("relative_record", False):
        F["rec_n_entities"] = tmp.groupby("r")["r"].transform("size").astype(np.float32).values
        for col in ("char_cos", "addr"):
            rk, gap, _ = _group_rank(tmp, "r", col)
            F[f"recrel_{col}_rank"], F[f"recrel_{col}_gap"] = rk.values, gap.values
    return F
