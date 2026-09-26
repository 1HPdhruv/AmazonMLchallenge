"""Phase-2 scale benchmarks on the REAL files (measured, not guessed).

    python tools/phase2_bench.py records --n 200000          # per-record cost of each stage
    python tools/phase2_bench.py retrieval --s1-frac 0.01 --pools 100000,200000,400000,800000

Stages use the production code paths (adapter parse, build_representations, CandidateGenerator).
Peak RSS is sampled by a background psutil thread; results go to reports/phase2_bench.json (appended).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.getcwd())
from src.data.adapters import CompetitionDataAdapter, _canonical  # noqa: E402
from src.pipeline.core import load_cfg  # noqa: E402
from src.preprocessing.tokenize import build_representations  # noqa: E402
from src.retrieval.union import CandidateGenerator  # noqa: E402

PROC = psutil.Process()


class Peak:
    """Context manager: wall time + peak RSS (sampled every 50 ms) + RSS delta."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def __enter__(self):
        self.rss0 = PROC.memory_info().rss
        self.peak = self.rss0
        self.stop = False
        self.t = threading.Thread(target=self._watch, daemon=True)
        self.t.start()
        self.t0 = time.perf_counter()
        return self

    def _watch(self):
        while not self.stop:
            self.peak = max(self.peak, PROC.memory_info().rss)
            time.sleep(0.05)

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        self.stop = True
        self.t.join()
        self.rss1 = PROC.memory_info().rss
        rec = {"stage": self.name, "seconds": round(self.dt, 2), "peak_rss_mb": round(self.peak / 2**20),
               "delta_rss_mb": round((self.rss1 - self.rss0) / 2**20), "avail_gb": round(psutil.virtual_memory().available / 2**30, 2)}
        self.log.append(rec)
        print(rec, flush=True)


def sample_lines(path, p, seed, cap=None, keep_ids=None):
    """Bernoulli line sample of a TSV (streamed); keep_ids: always keep these entity_ids too."""
    r = random.Random(seed)
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            if (keep_ids is not None and line.split("\t", 1)[0] in keep_ids) or r.random() < p:
                parts = line.rstrip("\r\n").split("\t")
                rows.append((parts + [""] * 4)[:4])
                if cap and len(rows) >= cap and keep_ids is None:
                    break
    return pd.DataFrame(rows, columns=header).astype(str)


def records_bench(a, base, log):
    ad = CompetitionDataAdapter(base["data"]["root"], base["normalization"]["address_parser"])
    per = a.n // 2
    with Peak("read_sample_s2+s3", log):
        s2 = sample_lines(ad.path("train", "source2"), per / 5034616, 1)
        s3 = sample_lines(ad.path("train", "source3"), per / 5285603, 2)
        raw = pd.concat([s2, s3], ignore_index=True)
    n = len(raw)
    with Peak("adapter_parse", log):
        recs = _canonical(raw, "record_id", base["normalization"]["address_parser"])
    with Peak("build_representations", log):
        R = build_representations(recs, base["normalization"])
    cg = CandidateGenerator(base["retrieval"]).fit(R.iloc[: min(50000, n)])   # IDF fit (cost measured separately)
    with Peak("encode_tfidf_and_keys", log):
        enc = cg.encode(R)
    nnz = {k: int(v.nnz) for k, v in enc.items() if hasattr(v, "nnz")}
    rep_mb = R.memory_usage(deep=True).sum() / 2**20
    out = {"n_records": n, "sparse_nnz": nnz, "representations_mb": round(rep_mb),
           "per_record_us": {r["stage"]: round(1e6 * r["seconds"] / n, 1) for r in log[-4:]},
           "per_record_rep_bytes": round(rep_mb * 2**20 / n)}
    total_records = 5034616 + 5285603
    out["extrapolated_full_train_pool"] = {
        "records": total_records,
        "seconds_single_core": {k: round(v * total_records / 1e6) for k, v in out["per_record_us"].items()},
        "representations_gb": round(out["per_record_rep_bytes"] * total_records / 2**30, 1),
        "sparse_gb_char_word": round(sum(nnz.values()) * 12 / n * total_records / 2**30, 1),
    }
    print(json.dumps(out, indent=1))
    return out


def retrieval_bench(a, base, log):
    ad = CompetitionDataAdapter(base["data"]["root"], base["normalization"]["address_parser"])
    norm = base["normalization"]
    with Peak("read_s1_sample", log):
        s1 = sample_lines(ad.path("train", "source1"), a.s1_frac, 3)
    ids = set(s1.entity_id)
    # true records of the sampled entities, so recall@k is measurable at every pool size
    truth = {}
    with open(ad.path("train", "train_ground_truth"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            e, _, m = line.rstrip("\r\n").partition("\t")
            if e in ids and m:
                truth[e] = m.split(",")
    true_ids = {r for v in truth.values() for r in v}
    E = build_representations(_canonical(s1, "entity_id", norm["address_parser"]), norm)
    results = []
    for pool in [int(x) for x in a.pools.split(",")]:
        pl = []
        with Peak(f"pool{pool}:read", pl):
            per = pool / 2
            s2 = sample_lines(ad.path("train", "source2"), per / 5034616, 11, keep_ids=true_ids)
            s3 = sample_lines(ad.path("train", "source3"), per / 5285603, 12, keep_ids=true_ids)
            raw = pd.concat([s2, s3], ignore_index=True)
        with Peak(f"pool{pool}:parse+represent", pl):
            R = build_representations(_canonical(raw, "record_id", norm["address_parser"]), norm)
        cfg = json.loads(json.dumps(base["retrieval"]))
        for ch in ("char", "word"):
            cfg[ch]["k"] = 100                      # retrieve deep once; recall@k computed for k <= 100
        cfg["address"]["k"] = 100
        with Peak(f"pool{pool}:fit+encode", pl):
            cg = CandidateGenerator(cfg).fit(E)
            ee, re_ = cg.encode(E), cg.encode(R)
        with Peak(f"pool{pool}:retrieve", pl):
            cand = cg.retrieve(ee, re_)
        rid = R.index.map(lambda i: None)  # placeholder to keep memory low
        rec_ids = raw["entity_id"].values
        ent_ids = s1["entity_id"].values
        cand["eid"] = ent_ids[cand.e.values]
        cand["rid"] = rec_ids[cand.r.values]
        cand["y"] = [r in set(truth.get(e, ())) for e, r in zip(cand.eid, cand.rid)]
        n_true = sum(len(v) for v in truth.values())
        rec_at = {}
        for ch in ("char", "word", "addr"):
            rk = cand[f"{ch}_rank"]
            rec_at[ch] = {k: round(float(cand.y[rk <= k].sum() / n_true), 4) for k in (5, 10, 20, 50, 100)}
        # union at the SHIPPED budgets (k=20 for char/word/addr, exact as configured)
        shipped = (cand.hit_exact == 1) | (cand.char_rank <= 20) | (cand.word_rank <= 20) | (cand.addr_rank <= 20)
        u20 = cand[shipped]
        res = {"pool": len(R), "s1": len(E), "true_pairs": n_true, "recall_at_k": rec_at,
               "union_recall_shipped_k20": round(float(u20.y.sum() / n_true), 4),
               "union_cands_per_entity_shipped": round(len(u20) / len(E), 1),
               "stages": pl, "addr_dropped_keys": getattr(cg.addr, "dropped_keys", None)}
        log.extend(pl)
        results.append(res)
        print(json.dumps({k: v for k, v in res.items() if k != "stages"}), flush=True)
        del R, cg, ee, re_, cand, raw, s2, s3
        if psutil.virtual_memory().available < 1.5 * 2**30:
            print("stopping: available RAM below 1.5 GB", flush=True)
            break
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["records", "retrieval"])
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--s1-frac", type=float, default=0.01)
    ap.add_argument("--pools", default="100000,200000,400000,800000")
    a = ap.parse_args()
    base = load_cfg("base")
    log = []
    out = records_bench(a, base, log) if a.mode == "records" else retrieval_bench(a, base, log)
    path = "reports/phase2_bench.json"
    prev = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    prev[a.mode] = {"result": out, "stages": log, "args": vars(a), "cpu_count": os.cpu_count(),
                    "total_ram_gb": round(psutil.virtual_memory().total / 2**30, 1)}
    json.dump(prev, open(path, "w", encoding="utf-8"), indent=2, default=str)


if __name__ == "__main__":
    main()
