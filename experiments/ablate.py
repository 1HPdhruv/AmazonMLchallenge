"""Ablation harness for the deep-improvement pass (one change at a time).

    python experiments/ablate.py --variant B0                      # run + store a variant
    python experiments/ablate.py --variant X --ref B0              # run X and compare to stored B0
    python experiments/ablate.py --variant X --final               # also score the untouched holdout2 tree

Protocol (pre-registered in reports/ablation_log.md):
  * model: configs/model_final.yaml (+ variant overrides); trained on train_fit; isotonic on calib
  * decision: configs/decision.yaml mechanism family (+ variant overrides), parameters tuned on calib
  * VAL  = entity-level validation split of the synthetic TRAIN tree (US/India)
  * DEV  = synthetic TEST tree data/synthetic (adds France) -> France development slice (hidden labels)
  * HOLDOUT2 = data/synthetic_holdout2, a fresh test tree touched only by --final
Keep rules:
  measurable change      : VAL paired-bootstrap P(improve) >= 0.9
  France-targeted change : DEV-France P(improve) >= 0.9 AND VAL P(improve) >= 0.2
  real-format bug fix    : unit tests pass AND VAL P(improve) >= 0.2 AND DEV-France P(improve) >= 0.2
SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.data.adapters import SyntheticDataAdapter  # noqa: E402
from src.decision.calibration import Calibrator  # noqa: E402
from src.decision.tournament import Frame, tune  # noqa: E402
from src.evaluation.entity_metrics import fbeta, paired_bootstrap, per_entity, stratified  # noqa: E402
from src.pipeline.core import Context, load_cfg, train_model  # noqa: E402
from src.pipeline.predict import run_inference  # noqa: E402
from src.pipeline.submit import CAND_COLS, MATCH_COLS, read_output  # noqa: E402

LABEL = "SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE"
OUT = "experiments/ablation"

# Each variant = overrides of base / model / decision configs. Variants accumulate by naming a parent.
VARIANTS: dict[str, dict] = {
    "B0": {"parent": None, "desc": "baseline: post-compliance code, model_final + calibrated multi"},
    # --- decision layer (audit #3, #4, #8)
    "D1": {"parent": "B0", "desc": "tune(): prefer the most conservative params on calib-F0.5 plateaus",
           "decision": {"tie_break": "high"}},
    "D2a": {"parent": "B0", "desc": "record exclusivity among accepted pairs (Source 1 deduplicated)",
            "decision": {"exclusive_mode": "accepted"}},
    "D2b": {"parent": "B0", "desc": "record exclusivity over ALL candidates, exact ties dropped",
            "decision": {"exclusive_mode": "all"}},
    "D3": {"parent": "D2b", "desc": "abs2: top-1 >= t1, secondaries >= absolute t2 (t2 may exceed t1)",
           "decision": {"mechanism": "abs2"}},
    # --- names (audit #1, #2): legal-form dictionary, trailing-span only, country-aware, generic fallback
    "N1": {"parent": "D2b", "desc": "legal_forms.yaml: trailing-span, per-country, canonical classes; "
                                    "char channel on legal-stripped name; das/da/do/die/der kept",
           "base": {"normalization": {"legal_dictionary": "configs/legal_forms.yaml"},
                    "retrieval": {"char_field": "name_fold_core"}}},
    "N1a": {"parent": "D2b", "desc": "legal_forms.yaml only (char channel still on the full folded name)",
            "base": {"normalization": {"legal_dictionary": "configs/legal_forms.yaml"}}},
    "N1c": {"parent": "D2b", "desc": "N1a + India 'company'/'and company' coverage fix (one re-run)",
            "base": {"normalization": {"legal_dictionary": "configs/legal_forms.yaml"}}},
    # --- France zero-shot (audit #2): unseen-token IDF policy
    "N2": {"parent": "D2b", "desc": "word IDF: unseen tokens -> train median IDF (not max) + unseen-count features",
           "base": {"retrieval": {"unseen_idf": {"word": "median"}}}, "model": {"features": {"unseen_counts": True}}},
    # --- addresses (audit #5, #9, #10): content-based parser v2
    "A1": {"parent": "D2b", "desc": "address parser v2: landmarks, unit segment, content-chosen street, postal "
                                    "anywhere, FR articles/types, CA/IN state guard",
           "base": {"normalization": {"address_parser": "v2"}}},
    # --- retrieval (audit #6): keep the whole tied block at the top-k cutoff instead of row-order cut
    "R1": {"parent": "A1", "desc": "top-k ties at the cutoff kept as a block (cap 50) in char/word/addr channels",
           "base": {"retrieval": {"tie_cap": 50}}},
    # --- retrieval (audit #5): parse-free address evidence
    "R2f": {"parent": "A1", "desc": "addr_text_cos FEATURE only (char TF-IDF over the full folded address)",
            "base": {"retrieval": {"addr_text_feature": True}}},
    "R2": {"parent": "A1", "desc": "addr_text CHANNEL (k=20, cos>=0.3) + addr_text_cos feature",
           "base": {"retrieval": {"addr_text_feature": True, "addr_text": {"enabled": True, "k": 20, "min_score": 0.3}}}},
    # --- retrieval (audit #7): chain saturation -> name top-100 re-scored by address similarity
    "R3": {"parent": "A1", "desc": "name_geo CHANNEL: char name top-100 re-scored name_cos + 1.0*addr_cos, keep 20",
           "base": {"retrieval": {"addr_text_feature": True,
                                  "name_geo": {"enabled": True, "k": 20, "k_wide": 100, "min_score": 0.3, "beta": 1.0}}}},
    # --- phonetic similarity (user item b): Metaphone + NYSIIS (jellyfish, MIT)
    "P1f": {"parent": "A1", "desc": "phonetic FEATURES: Metaphone/NYSIIS token-code Jaccard + code-set equality",
            "model": {"features": {"phonetic": True}}},
    "P1c": {"parent": "A1", "desc": "phonetic features + exact CHANNEL on sorted Metaphone code set (<=30 postings)",
            "model": {"features": {"phonetic": True}},
            "base": {"retrieval": {"phon": {"enabled": True, "max_postings": 30}}}},
    # --- real-format bug fix (audit #10): dotted initials / M/s honorific
    "B1": {"parent": "A1", "desc": "collapse dotted initials (S.K./M.G./L.L.C.) + strip M/s, Messrs (raw text)",
           "base": {"normalization": {"collapse_initials": True}}},
    # --- France zero-shot decision (audit decision#4): stricter threshold for countries unseen in train,
    #     margin estimated by a leave-one-country-out drill on TRAIN (never on dev/test)
    "F1": {"parent": "B1", "desc": "unseen-country threshold margin estimated by LOCO on train (US<->India)",
           "decision": {"unseen_margin": "loco"}},
}


def deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def resolve(name: str):
    chain, n = [], name
    while n:
        chain.append(VARIANTS[n])
        n = VARIANTS[n]["parent"]
    base, model, dec = load_cfg("base"), load_cfg("model_final"), load_cfg("decision")
    dec.setdefault("exclusive_mode", "accepted" if dec.get("record_exclusive") else "off")
    dec.setdefault("tie_break", "low")
    for v in reversed(chain):
        deep_update(base, v.get("base", {}))
        deep_update(model, v.get("model", {}))
        deep_update(dec, v.get("decision", {}))
    return base, model, dec


def loco_margin(ctx, mcfg, dec):
    """Leave-one-country-out drill on TRAIN only: train on one training country, tune the decision on that
    country's calib entities vs on the held-out country's train entities; the threshold shift needed for a
    country the model never saw is averaged over both directions. No test/dev data is touched."""
    from src.models.gbdt import PairModel, entity_weights
    from src.pipeline.core import feature_columns
    cols = feature_columns(ctx.F.columns, mcfg["features"])
    cc = ctx.attrs.set_index("e")["country"]
    countries = sorted(set(cc[np.concatenate([ctx.idx["train_fit"], ctx.idx["calib"]])]))
    shifts = []
    for seen in countries:
        other = [c for c in countries if c != seen]
        fit_e = np.array([e for e in ctx.idx["train_fit"] if cc[e] == seen])
        cal_seen = np.array([e for e in ctx.idx["calib"] if cc[e] == seen])
        unseen_e = np.array([e for e in np.concatenate([ctx.idx["train_fit"], ctx.idx["calib"]]) if cc[e] in other])
        rows = np.isin(ctx.cand["e"].values, fit_e)
        m = PairModel(mcfg["xgb"], mcfg["objective"]).fit(ctx.F.loc[rows, cols], ctx.y[rows],
                                                          entity_weights(ctx.cand["e"].values[rows], mcfg["entity_weighting"]))
        s = m.predict(ctx.F[cols])
        calr = np.isin(ctx.cand["e"].values, cal_seen)
        s = Calibrator().fit(s[calr], ctx.y[calr]).transform(s) if dec["calibrated"] else s
        fr = Frame(ctx.cand["e"].values, ctx.cand["r"].values, s, ctx.y)
        p_seen, _ = tune(fr, dec["mechanism"], cal_seen, ctx.n_true, dec["exclusive_mode"], dec["tie_break"])
        p_un, _ = tune(fr, dec["mechanism"], unseen_e, ctx.n_true, dec["exclusive_mode"], dec["tie_break"])
        key = "t" if "t" in p_seen else "t1"
        shifts.append(float(p_un[key]) - float(p_seen[key]))
    margin = max(0.0, float(np.mean(shifts)))
    print(f"LOCO shifts {dict(zip(countries, shifts))} -> unseen-country margin {margin:.3f}", file=sys.stderr)
    return margin


def tree_eval(system, model_path, root, tag):
    ad = SyntheticDataAdapter(root, parser=(system["base_cfg"].get("normalization") or {}).get("address_parser", "v1"))
    ents, recs, _ = ad.load("test")
    lab = ad.hidden_test_labels(set(ents.entity_id), set(recs.record_id))
    truth = lab.groupby("entity_id")["record_id"].apply(frozenset).to_dict()
    with tempfile.TemporaryDirectory() as out:
        paths, stats = run_inference(system, ents, recs, model_path, out)
        _, m = read_output(paths["matching_results.tsv"], MATCH_COLS)
        _, c = read_output(paths["candidate_pairs.tsv"], CAND_COLS)
    pred = {e: frozenset(v.split(",")) if v else frozenset() for e, v, _ in m}
    cand = {e: set(v.split(",")) if v else set() for e, v, _ in c}
    df = ents[["entity_id", "country"]].copy()
    df["f05"] = [fbeta(pred[e], truth.get(e, frozenset())) for e in df.entity_id]
    df["n_true"] = [len(truth.get(e, ())) for e in df.entity_id]
    df["n_pred"] = [len(pred[e]) for e in df.entity_id]
    df["tp"] = [len(pred[e] & truth.get(e, frozenset())) for e in df.entity_id]
    df["in_cand"] = [len(truth.get(e, frozenset()) & cand[e]) for e in df.entity_id]

    def summ(g):
        nm = g[g.n_true == 0]
        return {"n": int(len(g)), "macro_f05": round(float(g.f05.mean()), 4),
                "pair_precision": round(float(g.tp.sum() / max(1, g.n_pred.sum())), 4),
                "pair_recall": round(float(g.tp.sum() / max(1, g.n_true.sum())), 4),
                "no_match_fp": round(float((nm.n_pred > 0).mean()), 4) if len(nm) else None,
                "cand_recall": round(float(g.in_cand.sum() / max(1, g.n_true.sum())), 4)}
    res = {"overall": summ(df), **{f"country={k}": summ(g) for k, g in df.groupby("country")},
           "candidate_pairs": stats["candidate_pairs"]}
    return res, df[["entity_id", "country", "f05"]]


def run(name: str, final: bool):
    t0 = time.time()
    base, mcfg, dec = resolve(name)
    base["paths"]["reports"] = os.path.join(OUT, name, "reports")
    os.makedirs(base["paths"]["reports"], exist_ok=True)
    ctx = Context().prepare(base, mcfg)
    model, s, cols = train_model(ctx, mcfg)
    cal = ctx.rows("calib")
    calib = Calibrator().fit(s[cal], ctx.y[cal])
    sd = calib.transform(s) if dec["calibrated"] else s
    fr = Frame(ctx.cand["e"].values, ctx.cand["r"].values, sd, ctx.y)
    p, f_cal = tune(fr, dec["mechanism"], ctx.idx["calib"], ctx.n_true, dec["exclusive_mode"], dec["tie_break"])
    acc = fr.accept(dec["mechanism"], p, dec["exclusive_mode"])
    unseen_margin = loco_margin(ctx, mcfg, dec) if dec.get("unseen_margin") == "loco" else 0.0
    pe = per_entity(fr.df["e"].values, acc, ctx.y, ctx.idx["val"], ctx.n_true, ctx.cand_count).merge(ctx.attrs, on="e")
    val = stratified(pe, ("country", "true_sources"))
    tp_val = ctx.truth_pairs_for("val")
    val_recall = len(set(zip(ctx.cand.e.values, ctx.cand.r.values)) & tp_val) / len(tp_val)
    system = {"label": LABEL, "base_cfg": base, "model_cfg": mcfg,
              "decision": {"calibrated": dec["calibrated"], "mechanism": dec["mechanism"], "params": p,
                           "exclusive_mode": dec["exclusive_mode"], "unseen_margin": unseen_margin},
              "train_countries": sorted(set(ctx.E["country_n"].dropna())),
              "candgen": ctx.cg, "tidf": ctx.tidf, "fs": ctx.fs, "booster_features": cols, "calibrator": calib,
              "source_levels": list(pd.factorize(ctx.recs["source"])[1])}
    d = os.path.join(OUT, name)
    model.save(os.path.join(d, "model.json"))
    dev, dev_df = tree_eval(system, os.path.join(d, "model.json"), base["data"]["root"], "dev")
    out = {"label": LABEL, "variant": name, "desc": VARIANTS[name]["desc"],
           "decision_params": dict(p, unseen_margin=unseen_margin) if unseen_margin else p,
           "calib_f05": round(f_cal, 4), "val": val, "val_union_recall": round(val_recall, 4),
           "val_pairs": int(len(ctx.cand)), "dev": dev, "features": len(cols), "runtime_s": round(time.time() - t0, 1)}
    np.save(os.path.join(d, "val_vec.npy"), pe["f05"].values)
    dev_df.to_csv(os.path.join(d, "dev_vec.csv"), index=False)
    if final:
        ho, ho_df = tree_eval(system, os.path.join(d, "model.json"), "data/synthetic_holdout2", "holdout2")
        out["holdout2"] = ho
        ho_df.to_csv(os.path.join(d, "holdout2_vec.csv"), index=False)
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    return out


def compare(name: str, ref: str):
    a = np.load(os.path.join(OUT, ref, "val_vec.npy"))
    b = np.load(os.path.join(OUT, name, "val_vec.npy"))
    da = pd.read_csv(os.path.join(OUT, ref, "dev_vec.csv"))
    db = pd.read_csv(os.path.join(OUT, name, "dev_vec.csv"))
    m = da.merge(db, on=["entity_id", "country"], suffixes=("_a", "_b"))
    fr = m[m.country == "France"]
    return {"val": paired_bootstrap(a, b), "dev_all": paired_bootstrap(m.f05_a.values, m.f05_b.values),
            "dev_france": paired_bootstrap(fr.f05_a.values, fr.f05_b.values)}


def log_row(name, ref, res, cmp):
    path = "reports/ablation_log.md"
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Ablation log (deep-improvement pass)\n\n**{LABEL}**\n\n"
                    "Pre-registered keep rules: measurable change -> VAL P(improve) >= 0.9; France-targeted -> "
                    "DEV-France P(improve) >= 0.9 and VAL P(improve) >= 0.2; real-format bug fix -> unit tests + "
                    "VAL and DEV-France P(improve) >= 0.2. DEV = synthetic test tree (adds France); HOLDOUT2 is "
                    "touched only for the final report.\n\n"
                    "| variant | vs | description | VAL F0.5 | VAL no-match FP | VAL union recall | DEV F0.5 | DEV France F0.5 | "
                    "DEV France cand recall | pairs (val tree) | VAL bootstrap | DEV-France bootstrap | decision params |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    v, dv = res["val"]["overall"], res["dev"]
    frs = dv.get("country=France", {})
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"| {name} | {ref or '-'} | {res['desc']} | {v['macro_f05']} | {v['no_match_fp_rate']} | "
                f"{res['val_union_recall']} | {dv['overall']['macro_f05']} | {frs.get('macro_f05')} | "
                f"{frs.get('cand_recall')} | {res['val_pairs']} | {cmp['val'] if cmp else '-'} | "
                f"{cmp['dev_france'] if cmp else '-'} | {res['decision_params']} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    logging.disable(logging.INFO)
    res = run(a.variant, a.final)
    cmp = compare(a.variant, a.ref) if a.ref else None
    log_row(a.variant, a.ref, res, cmp)
    v = res["val"]["overall"]
    print(json.dumps({"variant": a.variant, "val_f05": v["macro_f05"], "val_nm_fp": v["no_match_fp_rate"],
                      "val_recall": res["val_union_recall"], "dev": res["dev"]["overall"],
                      "dev_france": res["dev"].get("country=France"), "holdout2": res.get("holdout2", {}).get("overall"),
                      "cmp": cmp, "params": res["decision_params"], "runtime_s": res["runtime_s"]}, default=float))


if __name__ == "__main__":
    main()
