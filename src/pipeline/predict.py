"""Inference on the OFFICIAL TEST files -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python -m src.pipeline.predict                  # uses artifacts/final/system.pkl + configs/base.yaml

Steps: load test sources (never labels) -> retrieval with train-fit statistics -> the candidate set
is written to candidate_pairs.tsv exactly as scored -> GBDT scores -> decision -> matching_results.tsv
-> local rule checks -> official validator (hard gate when present).
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.adapters import SyntheticDataAdapter, make_adapter
from src.decision.tournament import Frame
from src.features.pairs import build_features, fs_agreements
from src.pipeline.core import load_cfg, setup_logging
from src.pipeline.submit import (OutputContractError, build_lists, local_checks, official_gate, score_matching,
                                 write_outputs, write_report)
from src.preprocessing.address import norm_country
from src.preprocessing.tokenize import build_representations

log = logging.getLogger("er")


def score(system, ents: pd.DataFrame, recs: pd.DataFrame, model_path: str):
    """Candidate frame (exactly the pairs the model scores) with raw and decision-ready scores."""
    norm = system["base_cfg"].get("normalization")
    E = build_representations(ents.reset_index(drop=True), norm)
    R = build_representations(recs.reset_index(drop=True), norm)
    cg = system["candgen"]
    e_enc, r_enc = cg.encode(E), cg.encode(R)
    cand = cg.retrieve(e_enc, r_enc)
    levels = {s: i for i, s in enumerate(system["source_levels"])}
    src_codes = recs["source"].map(levels).fillna(-1).astype(int).values
    groups = {k: True for k in system["model_cfg"]["features"]}
    F = build_features(cand, E, R, e_enc, r_enc, system["tidf"], None, groups, src_codes)
    F["fs_weight"] = system["fs"].score(fs_agreements(F))
    booster = xgb.Booster()
    booster.load_model(model_path)
    cols = system["booster_features"]
    missing = [c for c in cols if c not in F]
    if missing:
        raise ValueError(f"features missing at inference: {missing}")
    s = booster.predict(xgb.DMatrix(F[cols].values, feature_names=cols, missing=np.nan))
    assert len(s) == len(cand), "every candidate pair must be scored exactly once"
    cand["s_raw"] = s
    cand["s"] = system["calibrator"].transform(s) if system["decision"]["calibrated"] else s
    return cand


def apply_decision(cand: pd.DataFrame, dec: dict) -> np.ndarray:
    fr = Frame(cand["e"].values, cand["r"].values, cand["s"].values, np.zeros(len(cand), np.int8))
    if dec["mechanism"] == "hybrid_by_bucket":
        cc = np.bincount(cand["e"].values)
        b = pd.cut(pd.Series(cc[cand["e"].values]), [-1, 1, 4, 20, 10 ** 9],
                   labels=["0-1", "2-4", "5-20", "20+"]).astype(str).values
        acc = np.zeros(len(cand), bool)
        for bucket, c in dec["params"].items():
            acc |= fr.accept(c["mechanism"], c["params"]) & (b == bucket)
        return acc
    mode = dec.get("exclusive_mode", "accepted" if dec.get("record_exclusive") else "off")
    return accept_with_unseen_margin(fr, dec, mode, cand.get("unseen_country"))


def accept_with_unseen_margin(fr, dec, mode, unseen_mask=None):
    """Entities whose country never occurs in train get a stricter threshold (t + unseen_margin),
    the margin estimated by a leave-one-country-out drill on TRAIN (experiments/ablate.py loco_margin)."""
    acc = fr.accept(dec["mechanism"], dec["params"], mode)
    m = float(dec.get("unseen_margin") or 0.0)
    if m and unseen_mask is not None and np.any(unseen_mask):
        p2 = {k: (min(0.999, v + m) if k in ("t", "t1") else v) for k, v in dec["params"].items()}
        acc2 = fr.accept(dec["mechanism"], p2, mode)
        acc = np.where(np.asarray(unseen_mask, bool), acc2, acc)
    return acc


def run_inference(system, ents, recs, model_path, out_dir):
    t0 = time.time()
    cand = score(system, ents, recs, model_path)
    train_cc = set(system.get("train_countries") or [])
    if train_cc:
        cc = ents["country"].map(norm_country).values
        cand["unseen_country"] = ~np.isin(cc[cand["e"].values], list(train_cc))
    acc = apply_decision(cand, system["decision"])
    eid, rid = ents["entity_id"].values, recs["record_id"].values
    cand_pairs = pd.DataFrame({"entity_id": eid[cand["e"].values], "record_id": rid[cand["r"].values]})
    match_pairs = cand_pairs[acc].assign(score=cand["s"].values[acc])
    s1_ids = list(eid)
    cand_lists, match_lists = build_lists(s1_ids, cand_pairs, match_pairs)   # asserts matches ⊆ candidates
    paths = write_outputs(out_dir, s1_ids, cand_lists, match_lists)
    stats = {"test_source1_entities": len(s1_ids), "test_records": len(rid),
             "countries_source1": ents["country"].fillna("<empty>").value_counts().to_dict(),
             "candidate_pairs": int(len(cand_pairs)), "entities_with_0_candidates": sum(not v for v in cand_lists.values()),
             "accepted_pairs": int(acc.sum()), "entities_with_matches": sum(bool(v) for v in match_lists.values()),
             "runtime_s": round(time.time() - t0, 1)}
    log.info("inference: %s", stats)
    return paths, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="artifacts/final/system.pkl")
    ap.add_argument("--base", default="base", help="config in configs/ whose data.root holds dataset/test")
    ap.add_argument("--allow-missing-validator", action="store_true",
                    help="REAL mode only: proceed although utils/validate_submission.py is absent (NOT submittable)")
    a = ap.parse_args()
    setup_logging("logs/predict.log")
    with open(a.system, "rb") as f:
        system = pickle.load(f)
    base = load_cfg(a.base)
    if base["mode"] != system["base_cfg"]["mode"]:
        raise SystemExit(f"system trained in mode={system['base_cfg']['mode']} but config mode={base['mode']}")
    base["normalization"] = system["base_cfg"].get("normalization")   # parse test exactly like train
    ad = make_adapter(base)
    ents, recs, _ = ad.load("test")                     # test labels do not exist (real) / are hidden (synthetic)
    out_dir = base["output"]["dir"]
    paths, stats = run_inference(system, ents, recs, os.path.join(os.path.dirname(a.system), "model.json"), out_dir)
    issues = local_checks(out_dir, ents["entity_id"].tolist(), set(recs["record_id"]))
    gate = {"ran": False, "passed": None}
    try:
        gate = official_gate(base, out_dir, a.allow_missing_validator)
    finally:
        syn = None
        if isinstance(ad, SyntheticDataAdapter):
            lab = ad.hidden_test_labels(set(ents.entity_id), set(recs.record_id))
            syn = score_matching(paths["matching_results.tsv"], lab, ents["entity_id"].tolist())
        write_report(os.path.join(base["paths"]["submissions"], "submission_validation_report.md"),
                     base["metric_label"], stats, issues, gate, syn)
    if issues:
        raise OutputContractError("local rule checks failed:\n" + "\n".join(issues))
    log.info("outputs OK: %s | official validator: %s%s", paths, gate,
             f" | {base['metric_label']}: macro F0.5 vs hidden synthetic labels = {syn:.4f}" if syn is not None else "")


if __name__ == "__main__":
    main()
