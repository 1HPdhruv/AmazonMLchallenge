"""Shared pipeline pieces used by train / predict / experiments."""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import pandas as pd
import yaml

from src.data.adapters import make_adapter
from src.data.splits import make_splits
from src.data.validation import validate, write_schema_report
from src.decision.calibration import Calibrator
from src.decision.tournament import Frame, tune
from src.evaluation.entity_metrics import per_entity, stratified
from src.features.pairs import FellegiSunter, TokenIdf, build_features, fs_agreements
from src.models.gbdt import PairModel, entity_weights
from src.preprocessing.tokenize import build_representations
from src.retrieval.union import CandidateGenerator, retrieval_report

log = logging.getLogger("er")


def load_cfg(name):
    with open(os.path.join("configs", f"{name}.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(path="logs/run.log"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()])


def feature_columns(all_cols, groups: dict):
    out = []
    for c in all_cols:
        if (c.startswith("rel_") or c == "cand_count") and not groups.get("relative_entity"):
            continue
        if (c.startswith("recrel_") or c == "rec_n_entities") and not groups.get("relative_record"):
            continue
        if c == "fs_weight" and not groups.get("fs"):
            continue
        if c.startswith("unseen_") and not groups.get("unseen_counts"):
            continue
        out.append(c)
    return out


class Context:
    """Everything that is fixed across model experiments: data, splits, candidates, features."""

    def prepare(self, base: dict, model_cfg: dict):
        t0 = time.time()
        self.base, self.label = base, base["metric_label"]
        ad = make_adapter(base)
        ents, recs, labels = ad.load("train")          # official TRAIN files only; test files never touched here
        self.schema = validate(ents, recs, labels, ad.present_fields, self.label, ad.issues)
        write_schema_report(self.schema, base["paths"]["reports"])
        self.ents, self.recs = ents.reset_index(drop=True), recs.reset_index(drop=True)
        self.E = build_representations(self.ents, base.get("normalization"))
        self.R = build_representations(self.recs, base.get("normalization"))
        self.eid = self.ents["entity_id"].values
        self.e_of = {e: i for i, e in enumerate(self.eid)}
        self.r_of = {r: i for i, r in enumerate(self.recs["record_id"].values)}
        self.src_codes = pd.factorize(self.recs["source"])[0]
        sp = base["splits"]
        self.split = make_splits(self.eid, base["seed"], sp["val_frac"], sp["test_frac"], sp["calib_frac_of_train"])
        self.split_arr = self.split.reindex(self.eid).values
        self.idx = {k: np.where(self.split_arr == k)[0] for k in ("train_fit", "calib", "val", "test")}
        # labels -> integer pairs
        self.labels = labels
        if labels is not None:
            lab = labels.drop_duplicates()
            self.true_pairs = set(zip(lab["entity_id"].map(self.e_of), lab["record_id"].map(self.r_of)))
            self.n_true = np.bincount([e for e, _ in self.true_pairs], minlength=len(self.eid))
        else:
            self.true_pairs, self.n_true = None, None
        # retrieval: statistics fit on train_fit entities only
        t1 = time.time()
        self.cg = CandidateGenerator(base["retrieval"]).fit(self.E.iloc[self.idx["train_fit"]])
        self.e_enc = self.cg.encode(self.E)
        self.r_enc = self.cg.encode(self.R)
        self.cand = self.cg.retrieve(self.e_enc, self.r_enc)
        self.retrieval_runtime = dict(self.cg.runtime, encode_s=round(time.time() - t1, 2))
        log.info("candidates: %d pairs for %d entities (%.1fs)", len(self.cand), len(self.eid), time.time() - t1)
        ce, cr = self.cand["e"].values, self.cand["r"].values
        self.y = np.array([(e, r) in self.true_pairs for e, r in zip(ce, cr)], np.int8) \
            if self.true_pairs is not None else np.zeros(len(ce), np.int8)
        self.cand_count = np.bincount(ce, minlength=len(self.eid))
        # features (all groups computed once; experiments select columns)
        t2 = time.time()
        self.tidf = TokenIdf(self.cg.word, model_cfg["idf_rare_quantile"], self.E.iloc[self.idx["train_fit"]]["name_clean"])
        all_groups = {k: True for k in model_cfg["features"]}
        self.F = build_features(self.cand, self.E, self.R, self.e_enc, self.r_enc, self.tidf, None, all_groups,
                                self.src_codes)
        tr = self.rows("train_fit")
        self.fs = FellegiSunter().fit(fs_agreements(self.F[tr]), self.y[tr])
        self.F["fs_weight"] = self.fs.score(fs_agreements(self.F))
        log.info("features: %s (%.1fs)", self.F.shape, time.time() - t2)
        # stratification attributes
        attrs = pd.DataFrame({"e": np.arange(len(self.eid)), "country": self.E["country_n"].fillna("NA").values})
        if labels is not None:
            src = labels.assign(e=labels["entity_id"].map(self.e_of),
                                s=labels["record_id"].map(dict(zip(self.recs.record_id, self.recs.source))))
            ts = src.groupby("e")["s"].apply(lambda x: "+".join(sorted(set(x))))
            attrs["true_sources"] = attrs["e"].map(ts).fillna("none")
        self.attrs = attrs
        self.prepare_s = round(time.time() - t0, 1)
        return self

    def rows(self, split):
        return np.isin(self.cand["e"].values, self.idx[split])

    def truth_pairs_for(self, split):
        s = set(self.idx[split])
        return {(e, r) for e, r in self.true_pairs if e in s}

    def write_retrieval_reports(self, out_dir):
        reps = {}
        for name, ents in (("retrieval_report", None), ("retrieval_report_val", self.idx["val"])):
            tp = self.true_pairs if ents is None else self.truth_pairs_for("val")
            reps[name] = retrieval_report(self.cand, len(self.eid), tp, self.retrieval_runtime, self.label,
                                          out_dir, name, entity_filter=ents)
        return reps


# ------------------------------------------------------------------ model experiment helpers
def train_model(ctx: Context, model_cfg: dict, extra_weight=None):
    cols = feature_columns(ctx.F.columns, model_cfg["features"])
    tr = ctx.rows("train_fit")
    X, y, e = ctx.F.loc[tr, cols], ctx.y[tr], ctx.cand["e"].values[tr]
    w = entity_weights(e, model_cfg["entity_weighting"])
    if extra_weight is not None:
        w = w * extra_weight
    m = PairModel(model_cfg["xgb"], model_cfg["objective"]).fit(X, y, w, groups=e)
    s = m.predict(ctx.F[cols])
    return m, s, cols


def reference_eval(ctx: Context, s, mechs=("top1_t", "multi"), calibrate=False, split="val"):
    """Tune each reference mechanism on calib, score on `split` (stratified)."""
    if calibrate:
        cal = ctx.rows("calib")
        s = Calibrator().fit(s[cal], ctx.y[cal]).transform(s)
    fr = Frame(ctx.cand["e"].values, ctx.cand["r"].values, s, ctx.y)
    out = {}
    for mech in mechs:
        p, f_cal = tune(fr, mech, ctx.idx["calib"], ctx.n_true)
        acc = fr.accept(mech, p)
        pe = per_entity(fr.df["e"].values, acc, ctx.y, ctx.idx[split], ctx.n_true, ctx.cand_count)
        pe = pe.merge(ctx.attrs, on="e", how="left")
        out[mech] = {"params": p, "calib_f05": round(f_cal, 4), "f05_vec": pe["f05"].values,
                     "strata": stratified(pe, extra_cols=("country", "true_sources"))}
    return out


def pair_diagnostics(ctx, s, split="val"):
    from sklearn.metrics import log_loss, roc_auc_score
    m = ctx.rows(split)
    yy, ss = ctx.y[m], np.clip(s[m], 1e-6, 1 - 1e-6)
    return {"pair_auc": round(float(roc_auc_score(yy, ss)), 4),
            "pair_logloss": round(float(log_loss(yy, ss)), 4) if s.max() <= 1 else None}


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)
