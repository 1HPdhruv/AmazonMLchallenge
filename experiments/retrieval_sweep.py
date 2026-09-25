"""Phase-3 retrieval sweep: candidate volume vs union recall, measured on TRAIN entities only
(train_fit + calib) so the chosen retrieval config never sees val/test labels.
SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE."""
from __future__ import annotations

import copy
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.data.adapters import make_adapter  # noqa: E402
from src.data.splits import make_splits  # noqa: E402
from src.pipeline.core import load_cfg  # noqa: E402
from src.preprocessing.tokenize import build_representations  # noqa: E402
from src.retrieval.union import CandidateGenerator, candidate_distribution  # noqa: E402


def main():
    base = load_cfg("base")
    ents, recs, labels = make_adapter(base).load("train")
    E, R = build_representations(ents), build_representations(recs)
    sp = base["splits"]
    split = make_splits(ents.entity_id, base["seed"], sp["val_frac"], sp["test_frac"], sp["calib_frac_of_train"])
    sa = split.reindex(ents.entity_id).values
    tr_fit = np.where(sa == "train_fit")[0]
    train = np.where(np.isin(sa, ["train_fit", "calib"]))[0]
    e_of = {e: i for i, e in enumerate(ents.entity_id)}
    r_of = {r: i for i, r in enumerate(recs.record_id)}
    tset = set(train)
    truth = {(e_of[a], r_of[b]) for a, b in zip(labels.entity_id, labels.record_id) if e_of[a] in tset}
    rows = []
    for k, ms, addr_k in itertools.product([10, 20], [0.2, 0.3, 0.4, 0.5], [10, 20]):
        cfg = copy.deepcopy(base["retrieval"])
        cfg["char"].update(k=k, min_score=ms)
        cfg["word"].update(k=k, min_score=ms)
        cfg["address"]["k"] = addr_k
        cg = CandidateGenerator(cfg).fit(E.iloc[tr_fit])
        cand = cg.retrieve(cg.encode(E), cg.encode(R))
        d = candidate_distribution(cand, len(E), truth, mask=cand["e"].isin(tset).values)
        counts = cand[cand.e.isin(tset)].e.value_counts().reindex(train, fill_value=0).values
        buckets = np.histogram(counts, [-1, 1, 4, 20, 10 ** 9])[0] / len(train)
        q = np.percentile(counts, [50, 95, 99])
        rows.append(dict(k=k, min_score=ms, addr_k=addr_k, recall=d["recall"], p50=float(q[0]), p95=float(q[1]),
                         p99=float(q[2]), max=int(counts.max()), mean=round(float(counts.mean()), 2),
                         bucket_frac={b: round(float(x), 3) for b, x in zip(["0-1", "2-4", "5-20", "20+"], buckets)}))
        print(rows[-1], flush=True)
    os.makedirs("reports", exist_ok=True)
    with open("reports/retrieval_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"label": base["metric_label"], "evaluated_on": "train_fit+calib entities", "rows": rows}, f, indent=2, ensure_ascii=False)
    lines = ["# Retrieval sweep (TRAIN entities only)\n", f"**{base['metric_label']}**\n",
             "| char/word k | min cosine | addr k | union recall | mean cands | P50 | P95 | P99 | max | bucket fractions |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['k']} | {r['min_score']} | {r['addr_k']} | {r['recall']} | {r['mean']} | {r['p50']} | "
                     f"{r['p95']} | {r['p99']} | {r['max']} | {r['bucket_frac']} |")
    with open("reports/retrieval_sweep.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
