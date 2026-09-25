"""Sparse retrieval engine (sparse_dot_topn), country partition + fallback, transliterated fields."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.preprocessing.tokenize import build_representations
from src.retrieval.tfidf import topk_cosine
from src.retrieval.union import CandidateGenerator


def _mats(seed=1):
    Q = normalize(sp.random(200, 3000, 0.01, format="csr", random_state=seed, dtype=np.float32))
    D = normalize(sp.random(2500, 3000, 0.01, format="csr", random_state=seed + 1, dtype=np.float32))
    return Q, D


def test_sparse_engine_matches_dense_exactly():
    Q, D = _mats()
    a = topk_cosine(Q, D, 20, 0.05, engine="dense")
    b = topk_cosine(Q, D, 20, 0.05, engine="sparse")
    A = {(int(x), int(y)): (round(float(s), 6), int(r)) for x, y, s, r in zip(*a)}
    B = {(int(x), int(y)): (round(float(s), 6), int(r)) for x, y, s, r in zip(*b)}
    assert A == B


def test_sparse_engine_respects_k_and_threshold():
    Q, D = _mats(3)
    qi, di, sc, rk = topk_cosine(Q, D, 5, 0.2, engine="sparse")
    assert (np.bincount(qi, minlength=Q.shape[0]) <= 5).all()
    assert (sc >= 0.2).all() and rk.min() >= 1


def _rep(names, countries):
    df = pd.DataFrame({"name": names, "address": None, "city": None, "region": None, "postal": None,
                       "country": countries, "full_address": None})
    return build_representations(df, {"address_parser": "v2", "transliterate": True})


def _cfg(**extra):
    cfg = {"channels": {"exact": False, "char": True, "word": False, "addr": False}, "exact_max_postings": 30,
           "char": {"k": 5, "min_score": 0.1}, "word": {"k": 5, "min_score": 0.1},
           "address": {"k": 5, "idf_floor": 3.0, "max_postings": 50}, "engine": "sparse"}
    cfg.update(extra)
    return cfg


def test_country_partition_keeps_cross_country_record_via_fallback():
    E = _rep(["Acme Traders", "Zenith Foods"], ["US", "India"])
    R = _rep(["Acme Traders", "Acme Tradrs", "Zenith Foods", "Zenith Food"], ["US", "India", "India", "US"])
    part = {"enabled": True, "fallback_k": 0}
    cg = CandidateGenerator(_cfg(partition=part)).fit(E)
    c0 = cg.retrieve(cg.encode(E), cg.encode(R))
    assert not ((c0.e == 0) & (c0.r == 1)).any()          # India record unreachable for a US entity without fallback
    part["fallback_k"] = 3
    cg = CandidateGenerator(_cfg(partition=part)).fit(E)
    c3 = cg.retrieve(cg.encode(E), cg.encode(R))
    assert ((c3.e == 0) & (c3.r == 1)).any()              # ...and reachable with the global fallback
    assert not c3.duplicated(["e", "r"]).any()


def test_transliterated_fields_bridge_scripts_for_retrieval_only():
    R = _rep(["హోటల్ మీడియా ప్రైవేట్ లిమిటెడ్", "Hotel Media Private Limited"], ["India", "India"])
    assert R.loc[0, "name_fold_tr"].isascii() and "limited" in R.loc[0, "name_fold_tr"]
    assert not R.loc[0, "name_fold"].isascii()            # original (feature) field untouched
    E = _rep(["Hotel Media Private Limited"], ["India"])
    cg = CandidateGenerator(_cfg(char_field="name_fold_tr")).fit(E)
    c = cg.retrieve(cg.encode(E), cg.encode(R))
    assert set(c.r) == {0, 1}                              # the Telugu-script record is now retrieved
