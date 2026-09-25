"""Train the final system from configs (model_final.yaml + decision.yaml) and save it.

    python -m src.pipeline.train

Fits retrieval statistics, FS weights, token IDF and the GBDT on train_fit; isotonic calibration
on the calib fold; decision parameters come from configs/decision.yaml (tuned on calib by the
oracle in run_experiments). Output: artifacts/final/system.pkl + model.json.
"""
from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import pandas as pd

from src.decision.calibration import Calibrator
from src.decision.tournament import Frame, tune
from src.models.hard_negative import mine, oof_scores
from src.models.gbdt import entity_weights
from src.pipeline.core import Context, feature_columns, load_cfg, setup_logging, train_model

log = logging.getLogger("er")


def resolve_decision(dcfg, ctx, calib, s):
    """exclusive_mode 'auto': Source 1 is officially deduplicated, so a record may belong to at most one
    Source-1 entity; enforce that (mode 'all') only if the TRAIN ground truth agrees (no record linked to
    more than one entity), else switch it off. retune: re-tune the mechanism's parameters on the calib
    fold for THIS model, so shipped parameters always match the trained scores."""
    dcfg = dict(dcfg)
    mode = dcfg.get("exclusive_mode", "accepted" if dcfg.get("record_exclusive") else "off")
    if mode == "auto":
        multi = (ctx.schema.get("labels") or {}).get("records_linked_to_multiple_entities")
        mode = "all" if multi == 0 else "off"
        log.info("exclusive_mode auto -> %s (records linked to >1 entity in train GT: %s)", mode, multi)
    dcfg["exclusive_mode"] = mode
    if dcfg.get("retune"):
        sd = calib.transform(s) if dcfg["calibrated"] else s
        fr = Frame(ctx.cand["e"].values, ctx.cand["r"].values, sd, ctx.y)
        p, f = tune(fr, dcfg["mechanism"], ctx.idx["calib"], ctx.n_true, mode, dcfg.get("tie_break", "low"))
        dcfg["params"] = {k: float(v) for k, v in p.items()}
        log.info("decision re-tuned on calib: %s %s (calib F0.5 %.4f)", dcfg["mechanism"], dcfg["params"], f)
    return dcfg


def build_system(base, mcfg, dcfg, ctx=None):
    ctx = ctx or Context().prepare(base, mcfg)
    extra = None
    if mcfg.get("hard_negative", {}).get("enabled"):
        tr = ctx.rows("train_fit")
        cols = feature_columns(ctx.F.columns, mcfg["features"])
        e_tr = ctx.cand["e"].values[tr]
        w = entity_weights(e_tr, mcfg["entity_weighting"])
        oof = oof_scores(ctx.F.loc[tr, cols], ctx.y[tr], w, e_tr, mcfg["xgb"], mcfg["hard_negative"]["folds"])
        extra, _ = mine(ctx.F[tr].reset_index(drop=True), ctx.y[tr], oof, e_tr, ctx.n_true[e_tr] > 0, mcfg["hard_negative"])
        assert set(np.unique(e_tr[extra > 1])) <= set(ctx.idx["train_fit"])
    model, s, cols = train_model(ctx, mcfg, extra_weight=extra)
    cal = ctx.rows("calib")
    calib = Calibrator().fit(s[cal], ctx.y[cal])
    dcfg = resolve_decision(dcfg, ctx, calib, s)
    system = {"label": base["metric_label"], "base_cfg": base, "model_cfg": mcfg, "decision": dcfg,
              "train_countries": sorted(set(ctx.E["country_n"].dropna())),
              "candgen": ctx.cg, "tidf": ctx.tidf, "fs": ctx.fs, "booster_features": cols, "calibrator": calib,
              "source_levels": list(pd.factorize(ctx.recs["source"])[1])}
    return system, model, ctx, s


def main():
    setup_logging("logs/train.log")
    base, dcfg = load_cfg("base"), load_cfg("decision")
    mcfg = load_cfg("model_final") if os.path.exists("configs/model_final.yaml") else load_cfg("model")
    system, model, ctx, _ = build_system(base, mcfg, dcfg)
    out = os.path.join(base["paths"]["artifacts"], "final")
    os.makedirs(out, exist_ok=True)
    model.save(os.path.join(out, "model.json"))
    with open(os.path.join(out, "system.pkl"), "wb") as f:
        pickle.dump(system, f)
    log.info("saved system to %s (decision=%s)", out, system["decision"])


if __name__ == "__main__":
    main()
