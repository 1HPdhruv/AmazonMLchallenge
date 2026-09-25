"""Exact-key and address retrieval channels.

Address channel: rarity-weighted blocking keys. Key IDF is fit on TRAIN entities only; keys with
IDF below `idf_floor` never fire, and a key whose postings list exceeds `max_postings` is dropped
(a structural runtime guard against candidate explosion, not a learned statistic). A bare house
number is never a key on its own — it is always combined with a street or city token.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.preprocessing.tokenize import ok


def resolve_cap(cap, pool_size: int) -> int:
    """A posting cap is either a fixed int (legacy) or {min: m, per_million: x}: max(m, x * pool/1e6).
    Fixed caps sized for a 12k-record synthetic pool silently drop most keys in a 10M-record pool."""
    if isinstance(cap, dict):
        return int(max(cap.get("min", 0), round(cap.get("per_million", 0) * pool_size / 1e6)))
    return int(cap)


def exact_channel(ent_keys, rec_keys, max_postings):
    post = defaultdict(list)
    for j, k in enumerate(rec_keys):
        if ok(k):
            post[k].append(j)
    max_postings = resolve_cap(max_postings, len(rec_keys))
    qi, di = [], []
    for i, k in enumerate(ent_keys):
        p = post.get(k) if ok(k) else None
        if p and len(p) <= max_postings:
            qi.extend([i] * len(p))
            di.extend(p)
    qi, di = np.array(qi, int), np.array(di, int)
    return qi, di, np.ones(len(qi)), np.ones(len(qi), int)


def address_keys(rep_row) -> list[str]:
    keys = []
    h, core, city, pc = (v if ok(v) else None for v in
                         (rep_row["house"], rep_row["street_core"], rep_row["city_n"], rep_row["postal_n"]))
    first = core.split()[0] if core else None
    if pc and len(pc) >= 4:
        keys.append(f"p|{pc}")
    if h and first:
        keys.append(f"hs|{h}|{first}")
    if h and city:
        keys.append(f"hc|{h}|{city}")
    if core and city:
        keys.append(f"sc|{core}|{city}")
    if pc and first:
        keys.append(f"ps|{pc}|{first}")
    return keys


class AddressChannel:
    def __init__(self, idf_floor: float, max_postings: int, k: int, tie_cap: int = 0):
        self.idf_floor, self.max_postings, self.k, self.tie_cap = idf_floor, max_postings, k, tie_cap
        self.df, self.n = {}, 0

    def fit(self, train_key_lists):
        self.n = len(train_key_lists)
        df = defaultdict(int)
        for ks in train_key_lists:
            for key in sorted(set(ks)):
                df[key] += 1
        self.df = dict(df)
        return self

    def idf(self, key):
        return float(np.log((1.0 + self.n) / (1.0 + self.df.get(key, 0))) + 1.0)

    def retrieve(self, ent_key_lists, rec_key_lists):
        post = defaultdict(list)
        for j, ks in enumerate(rec_key_lists):
            for key in sorted(set(ks)):
                post[key].append(j)
        self.dropped_keys = 0
        self.cap_used = cap = resolve_cap(self.max_postings, len(rec_key_lists))
        qi, di, sc, rk = [], [], [], []
        for i, ks in enumerate(ent_key_lists):
            acc = defaultdict(float)
            for key in sorted(set(ks)):  # sorted: hash-seed-independent float sums and tie order
                w = self.idf(key)
                p = post.get(key)
                if not p or w < self.idf_floor:
                    continue
                if len(p) > cap:
                    self.dropped_keys += 1
                    continue
                for j in p:
                    acc[j] += w
            if not acc:
                continue
            items = sorted(acc.items(), key=lambda x: (-x[1], x[0]))
            if self.tie_cap and len(items) > self.k:   # keep the whole block tied with the k-th score
                kth = items[self.k - 1][1]
                items = [it for it in items if it[1] >= kth - 1e-12][:max(self.tie_cap, self.k)]
            else:
                items = items[: self.k]
            qi.extend([i] * len(items))
            di.extend(j for j, _ in items)
            sc.extend(s for _, s in items)
            rk.extend(range(1, len(items) + 1))
        return np.array(qi, int), np.array(di, int), np.array(sc, float), np.array(rk, int)
