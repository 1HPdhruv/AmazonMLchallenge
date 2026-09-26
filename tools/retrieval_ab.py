"""Retrieval A/B on the fixed REAL-data slice used in Phase 2 (same seeds -> same 175k-record pool).

    python tools/retrieval_ab.py --variants V0,V1,V2,V3,V3a

The slice: 1% random Source-1 train entities (seed 3) x a pool of 100k random S2/S3 train records
(seeds 11/12) plus ALL true records of the sampled entities, so recall is measurable. Representations
are built once; each variant only changes retrieval. Results -> reports/retrieval_ab.json (+ .md).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.getcwd())
from src.data.adapters import CompetitionDataAdapter, _canonical  # noqa: E402
from src.pipeline.core import load_cfg  # noqa: E402
from src.preprocessing.tokenize import build_representations  # noqa: E402
from src.retrieval.union import CHANNELS, CandidateGenerator  # noqa: E402
from tools.phase2_bench import Peak, sample_lines  # noqa: E402

VARIANTS = {
    "V0": {"desc": "legacy dense engine (shipped)", "retrieval": {"engine": "dense"}},
    "V1": {"desc": "sparse_dot_topn engine (same config otherwise)", "retrieval": {"engine": "sparse"}},
    "V2": {"desc": "V1 + transliterated name fields (anyascii) for char/word/exact",
           "retrieval": {"engine": "sparse", "char_field": "name_fold_tr", "word_field": "name_clean_tr",
                         "key_field": "name_key_tr"}},
    "V3": {"desc": "V2 + country partition, global fallback top-3",
           "retrieval": {"engine": "sparse", "char_field": "name_fold_tr", "word_field": "name_clean_tr",
                         "key_field": "name_key_tr", "partition": {"enabled": True, "fallback_k": 3}}},
    "V3a": {"desc": "V2 + country partition, NO fallback",
            "retrieval": {"engine": "sparse", "char_field": "name_fold_tr", "word_field": "name_clean_tr",
                          "key_field": "name_key_tr", "partition": {"enabled": True, "fallback_k": 0}}},
}


def non_latin(s) -> bool:
    return isinstance(s, str) and any(ord(ch) > 0x24F and ch.isalpha() for ch in s)


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def build_slice(base):
    norm = dict(base["normalization"], transliterate=True)
    ad = CompetitionDataAdapter(base["data"]["root"], norm["address_parser"])
    s1 = sample_lines(ad.path("train", "source1"), 0.01, 3)
    ids = set(s1.entity_id)
    truth = {}
    with open(ad.path("train", "train_ground_truth"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            e, _, m = line.rstrip("\r\n").partition("\t")
            if e in ids and m:
                truth[e] = m.split(",")
    true_ids = {r for v in truth.values() for r in v}
    s2 = sample_lines(ad.path("train", "source2"), 50000 / 5034616, 11, keep_ids=true_ids)
    s3 = sample_lines(ad.path("train", "source3"), 50000 / 5285603, 12, keep_ids=true_ids)
    raw = pd.concat([s2.assign(src="S2"), s3.assign(src="S3")], ignore_index=True)
    E = build_representations(_canonical(s1, "entity_id", norm["address_parser"]), norm)
    R = build_representations(_canonical(raw.drop(columns="src"), "record_id", norm["address_parser"]), norm)
    pairs = pd.DataFrame([(e, r) for e, v in truth.items() for r in v], columns=["eid", "rid"])
    return s1, raw, E, R, pairs


def evaluate(cand, s1, raw, pairs):
    eids, rids = s1.entity_id.values, raw.entity_id.values
    got = pd.DataFrame({"eid": eids[cand.e.values], "rid": rids[cand.r.values]})
    for ch in [c for c in CHANNELS if f"hit_{c}" in cand]:
        got[ch] = cand[f"hit_{ch}"].values
    m = pairs.merge(got, on=["eid", "rid"], how="left")
    m["found"] = m[CHANNELS[0]].notna()
    ent_c = dict(zip(s1.entity_id, s1.country))
    rec_c = dict(zip(raw.entity_id, raw.country))
    rec_src = dict(zip(raw.entity_id, raw.src))
    e_name = dict(zip(s1.entity_id, s1.business_name))
    r_name = dict(zip(raw.entity_id, raw.business_name))
    m["e_country"] = m.eid.map(ent_c)
    m["src"] = m.rid.map(rec_src)
    m["cross_country"] = m.rid.map(rec_c) != m.e_country
    m["cross_script"] = [non_latin(e_name.get(e)) != non_latin(r_name.get(r)) for e, r in zip(m.eid, m.rid)]
    m["any_non_latin"] = [non_latin(e_name.get(e)) or non_latin(r_name.get(r)) for e, r in zip(m.eid, m.rid)]
    counts = cand.e.value_counts().reindex(range(len(s1)), fill_value=0).values
    q = np.percentile(counts, [50, 95, 99])

    def rec(mask):
        sub = m[mask]
        return {"n": int(len(sub)), "recall": round(float(sub.found.mean()), 4) if len(sub) else None}
    out = {
        "true_pairs": int(len(m)),
        "union_recall": round(float(m.found.mean()), 4),
        "channel_recall": {ch: round(float((m[ch] == 1).mean()), 4) for ch in CHANNELS if ch in m},
        "cands_per_entity": {"mean": round(float(counts.mean()), 1), "p50": float(q[0]), "p95": float(q[1]),
                             "p99": float(q[2]), "max": int(counts.max())},
        "candidate_pairs": int(len(cand)),
        "by_source": {s: rec(m.src == s) for s in ("S2", "S3")},
        "by_country": {c: rec(m.e_country == c) for c in sorted(m.e_country.dropna().unique())},
        "cross_country_pairs": rec(m.cross_country),
        "cross_script_pairs": rec(m.cross_script),
        "any_non_latin_pairs": rec(m.any_non_latin),
    }
    return out, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="V0,V1,V2,V3,V3a")
    a = ap.parse_args()
    base = load_cfg("base")
    log = []
    with Peak("build_slice", log):
        s1, raw, E, R, pairs = build_slice(base)
    print(f"slice: S1={len(E)} pool={len(R)} true_pairs={len(pairs)}", flush=True)
    results, cand_sets = {}, {}
    for v in a.variants.split(","):
        if psutil.virtual_memory().available < 2.0 * 2**30:
            print(f"skip {v}: available RAM below 2 GB", flush=True)
            continue
        cfg = deep_update(copy.deepcopy(base["retrieval"]), VARIANTS[v]["retrieval"])
        vlog = []
        with Peak(f"{v}:fit+encode", vlog):
            cg = CandidateGenerator(cfg).fit(E)
            ee, re_ = cg.encode(E), cg.encode(R)
        with Peak(f"{v}:retrieve", vlog):
            cand = cg.retrieve(ee, re_)
        out, got = evaluate(cand, s1, raw, pairs)
        out.update(desc=VARIANTS[v]["desc"], runtime_s={x["stage"]: x["seconds"] for x in vlog},
                   peak_rss_mb=max(x["peak_rss_mb"] for x in vlog), channel_runtime_s=dict(cg.runtime))
        cand_sets[v] = set(zip(got.eid, got.rid))
        results[v] = out
        print(v, json.dumps(out), flush=True)
        del cg, ee, re_, cand, got
    if "V0" in cand_sets and "V1" in cand_sets:
        results["V0_vs_V1_identical_candidate_sets"] = cand_sets["V0"] == cand_sets["V1"]
        results["V0_vs_V1_symmetric_difference"] = len(cand_sets["V0"] ^ cand_sets["V1"])
    results["_slice"] = {"s1": len(E), "pool": len(R), "true_pairs": len(pairs), "build": log}
    json.dump(results, open("reports/retrieval_ab.json", "w", encoding="utf-8"), indent=2, default=str)
    print("identical V0/V1:", results.get("V0_vs_V1_identical_candidate_sets"), results.get("V0_vs_V1_symmetric_difference"))


if __name__ == "__main__":
    main()
