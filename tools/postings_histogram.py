"""Posting-list size histogram for the exact-name and address-key channels on the REAL record pool.

    python tools/postings_histogram.py --split train --workers 3

Streams S2+S3 in chunks through the production key functions (adapter parse v2 -> build_representations
-> name_key / address_keys), maps each key to a stable 64-bit BLAKE2 hash (Python's hash() is salted per
process), counts postings with numpy, and reports per key type: posting-size quantiles and the share of
record-key occurrences that a max_postings cap of C would silently drop. Writes reports/postings_histogram.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

TYPES = ["name", "p", "hs", "hc", "sc", "ps"]


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


def chunk_keys(args):
    lines, header, norm = args
    from src.data.adapters import _canonical
    from src.preprocessing.tokenize import build_representations, ok
    from src.retrieval.channels import address_keys
    df = pd.DataFrame([(ln.rstrip("\r\n").split("\t") + [""] * 4)[:4] for ln in lines], columns=header).astype(str)
    R = build_representations(_canonical(df, "record_id", norm["address_parser"]), norm)
    out = {t: [] for t in TYPES}
    for k in R["name_key"].values:
        if ok(k):
            out["name"].append(_h(k))
    for _, row in R[["house", "street_core", "city_n", "postal_n"]].iterrows():
        for key in set(address_keys(row)):
            out[key.split("|", 1)[0]].append(_h(key))
    return {t: np.array(v, dtype=np.int64) for t, v in out.items()}


def reader(paths, header_ref, chunk, norm):
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            header_ref[:] = header
            buf = []
            for line in f:
                buf.append(line)
                if len(buf) >= chunk:
                    yield buf, header, norm
                    buf = []
            if buf:
                yield buf, header, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=100000)
    a = ap.parse_args()
    from src.data.adapters import CompetitionDataAdapter
    from src.pipeline.core import load_cfg
    base = load_cfg("base")
    ad = CompetitionDataAdapter(base["data"]["root"])
    paths = [ad.path(a.split, "source2"), ad.path(a.split, "source3")]
    t0 = time.time()
    acc = {t: [] for t in TYPES}
    n_chunks = 0
    with Pool(a.workers) as pool:
        for res in pool.imap(chunk_keys, reader(paths, [], a.chunk, base["normalization"]), chunksize=1):
            for t in TYPES:
                acc[t].append(res[t])
            n_chunks += 1
            if n_chunks % 10 == 0:
                print(f"{n_chunks * a.chunk:,} records, {time.time() - t0:.0f}s", flush=True)
    report = {"split": a.split, "records_pool": None, "runtime_s": None, "types": {}}
    caps = [30, 50, 100, 200, 500, 1000, 5000]
    for t in TYPES:
        h = np.concatenate(acc[t]) if acc[t] else np.zeros(0, np.int64)
        acc[t] = None
        _, counts = np.unique(h, return_counts=True)
        occ = counts.sum()
        q = np.percentile(counts, [50, 90, 99, 99.9]) if len(counts) else [0] * 4
        report["types"][t] = {
            "distinct_keys": int(len(counts)), "record_key_occurrences": int(occ),
            "posting_p50": float(q[0]), "posting_p90": float(q[1]), "posting_p99": float(q[2]),
            "posting_p99_9": float(q[3]), "posting_max": int(counts.max()) if len(counts) else 0,
            "share_of_occurrences_dropped_at_cap": {c: round(float(counts[counts > c].sum() / max(occ, 1)), 4) for c in caps},
            "share_of_keys_dropped_at_cap": {c: round(float((counts > c).mean()), 5) for c in caps},
        }
        print(t, report["types"][t], flush=True)
    report["records_pool"] = int(report["types"]["name"]["record_key_occurrences"])
    report["runtime_s"] = round(time.time() - t0)
    json.dump(report, open("reports/postings_histogram.json", "w", encoding="utf-8"), indent=2)
    print("runtime_s", report["runtime_s"])


if __name__ == "__main__":
    main()
