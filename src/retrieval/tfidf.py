"""Char / word TF-IDF retrieval with IDF fit on TRAIN entities only.

Hashing (no vocabulary) + an IDF table fit on train documents: a token never seen in train gets
the maximum IDF (df=0), so rare names of val/test entities stay rare without the IDF ever
seeing val/test text. This is the train-only rule of CONFIG.md, enforced structurally.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

N_FEATURES = 2 ** 20


class HashedTfidf:
    def __init__(self, kind: str, unseen: str = "max"):
        """unseen: IDF given to hashed features never seen in train. 'max' (df=0 -> rarest) or
        'median' (median IDF of the features seen in train; a train-only statistic). 'median' stops
        generic words of an unseen country from counting as brand-strength evidence."""
        self.kind = kind
        self.unseen = unseen
        if kind == "char":
            self.hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=N_FEATURES,
                                        alternate_sign=False, norm=None, lowercase=False)
        else:
            self.hv = HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=N_FEATURES,
                                        alternate_sign=False, norm=None, lowercase=False)
        self.idf = None
        self.n_fit_docs = 0

    def fit(self, train_docs):
        X = self.hv.transform(list(train_docs))
        df = np.bincount(X.indices, minlength=N_FEATURES)  # rows have unique indices -> doc freq
        self.n_fit_docs = X.shape[0]
        self.idf = np.log((1.0 + self.n_fit_docs) / (1.0 + df)) + 1.0
        self.seen = df > 0
        if self.unseen == "median" and self.seen.any():
            self.idf[~self.seen] = float(np.median(self.idf[self.seen]))
        return self

    def n_unseen(self, tokens):
        if not tokens or not hasattr(self, "seen"):
            return 0
        X = self.hv.transform([" ".join(tokens)])
        return int((~self.seen[X.indices]).sum())

    def transform(self, docs):
        X = self.hv.transform(list(docs)).astype(np.float32)
        X.data = 1.0 + np.log(X.data)  # sublinear tf
        X = X @ sp.diags(self.idf.astype(np.float32))
        return normalize(X, norm="l2", copy=False).tocsr()

    def token_idf(self, tokens):
        """IDF lookup for word tokens (used by IDF features)."""
        if not tokens:
            return np.zeros(0)
        X = self.hv.transform([" ".join(tokens)])
        return self.idf[X.indices]


def topk_cosine_sparse(Q, D, k: int, min_score: float, chunk: int = 20000, n_threads: int = 0, DT=None):
    """Sparse top-n cosine via sparse_dot_topn (Apache-2.0): never materialises a dense query x pool block.
    Cost scales with the shared non-zeros, not with pool size. Same return contract as topk_cosine."""
    from sparse_dot_topn import sp_matmul_topn
    if DT is None:
        DT = D.T.tocsr()
    n_threads = n_threads or max(1, (os.cpu_count() or 2) - 1)
    qi, di, sc, rk = [], [], [], []
    for s in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[s:s + chunk], DT, top_n=k, threshold=min_score, sort=True, n_threads=n_threads)
        C = C.tocsr()
        counts = np.diff(C.indptr)
        rows = np.repeat(np.arange(C.shape[0]), counts)
        keep = C.data >= min_score
        ranks = np.arange(len(C.data)) - np.repeat(C.indptr[:-1], counts) + 1   # data sorted desc per row
        qi.append(rows[keep] + s)
        di.append(C.indices[keep])
        sc.append(C.data[keep])
        rk.append(ranks[keep])
    if not qi:
        return (np.array([], int),) * 2 + (np.array([]), np.array([], int))
    return (np.concatenate(qi).astype(np.int64), np.concatenate(di).astype(np.int64),
            np.concatenate(sc), np.concatenate(rk).astype(np.int64))


def topk_cosine(Q, D, k: int, min_score: float, chunk: int = 1000, tie_cap: int = 0, engine: str = "dense",
                n_threads: int = 0, DT=None):
    """engine 'dense' = legacy dense-block path (small pools / tie_cap); 'sparse' = sparse_dot_topn."""
    if engine == "sparse" and not tie_cap:
        return topk_cosine_sparse(Q, D, k, min_score, n_threads=n_threads, DT=DT)
    return _topk_cosine_dense(Q, D, k, min_score, chunk, tie_cap)


def _topk_cosine_dense(Q, D, k: int, min_score: float, chunk: int = 1000, tie_cap: int = 0):
    """Returns (q_idx, d_idx, score, rank) for top-k cosine neighbours per query row.
    tie_cap > 0: records tied with the k-th score are ALL kept (up to tie_cap in total) instead of being
    cut by row position, which otherwise systematically favours records listed first (Source 2)."""
    qi, di, sc, rk = [], [], [], []
    DT = D.T.tocsc()
    for s in range(0, Q.shape[0], chunk):
        S = (Q[s:s + chunk] @ DT).toarray()
        kk = min(k, S.shape[1])
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        vals = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        for r in range(idx.shape[0]):
            if tie_cap and kk < S.shape[1]:
                kth = vals[r][kk - 1]
                if kth >= min_score and (S[r] >= kth - 1e-6).sum() > kk:
                    cand = np.where(S[r] >= kth - 1e-6)[0]
                    cand = cand[np.lexsort((cand, -S[r][cand]))][:max(tie_cap, kk)]
                    idx_r, vals_r = cand, S[r][cand]
                    keep = vals_r >= min_score
                    n = int(keep.sum())
                    qi.append(np.full(n, s + r))
                    di.append(idx_r[keep])
                    sc.append(vals_r[keep])
                    rk.append(np.arange(1, n + 1))
                    continue
            keep = vals[r] >= min_score
            n = int(keep.sum())
            if n:
                qi.append(np.full(n, s + r))
                di.append(idx[r][keep])
                sc.append(vals[r][keep])
                rk.append(np.arange(1, n + 1))
    if not qi:
        return (np.array([], int),) * 2 + (np.array([]), np.array([], int))
    return np.concatenate(qi), np.concatenate(di), np.concatenate(sc), np.concatenate(rk)
