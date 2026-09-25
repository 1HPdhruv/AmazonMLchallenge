"""Staged experiment matrix (Section 14) + decision-layer oracle (Section 9) + final holdout.

Stages (each keeps the winner and moves on; no full factorial):
  E1 baseline  -> E2 entity weighting -> E2r ranking comparison point -> E3 hard negatives
  -> E4 calibration + decision oracle -> E5 feature ablations (FS / candidate-relative)
  -> (oracle re-run if the feature set changed) -> FINAL on the test split.
Selection signal: entity macro F0.5 on VAL, with decision parameters tuned on CALIB.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd

from src.decision.calibration import Calibrator
from src.decision.tournament import Frame, low_count_simulation, run_oracle
from src.evaluation.entity_metrics import paired_bootstrap, per_entity, stratified
from src.evaluation.error_analysis import classify, diagnostic_tree, score_analysis
from src.models.hard_negative import mine, oof_scores
from src.pipeline.core import (Context, dump, feature_columns, load_cfg, pair_diagnostics, reference_eval,
                               setup_logging, train_model)

log = logging.getLogger("er")
REF = "multi"            # reference decision used for model-stage A/B (tuned on calib)
P_KEEP = 0.90            # keep a change only if paired-bootstrap P(improvement) on VAL entities >= this


class Log:
    def __init__(self, label):
        self.rows, self.label = [], label

    def add(self, **kw):
        self.rows.append(kw)
        log.info("EXPERIMENT %s: %s", kw.get("id"), {k: v for k, v in kw.items() if k != "id"})

    def write(self, path="EXPERIMENT_LOG.md"):
        cols = ["id", "hypothesis", "config", "candidate_recall_val", "val_macro_f05", "val_precision",
                "val_recall", "val_no_match_fp", "val_multi_p/r", "decision", "runtime_s", "diagnosis", "outcome"]
        out = [f"# Experiment log\n\n**{self.label}** — every number below is from the synthetic benchmark.\n",
               "Decision parameters are tuned on the calib fold (carved from train); every metric is on the VAL "
               "split unless the ID says TEST.\n",
               "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in self.rows:
            out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/").replace("\n", " ") for c in cols) + " |")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")


def _row(ev, mech=REF):
    o = ev[mech]["strata"]["overall"]
    return dict(val_macro_f05=o["macro_f05"], val_precision=o["pair_precision"], val_recall=o["pair_recall"],
                val_no_match_fp=o["no_match_fp_rate"], **{"val_multi_p/r": f"{o['multi_precision']}/{o['multi_recall']}"},
                decision=f"{mech} {ev[mech]['params']}")


def _f(ev, mech=REF):
    return ev[mech]["strata"]["overall"]["macro_f05"]


def _bt(ev_a, ev_b, mech=REF):
    return paired_bootstrap(ev_a[mech]["f05_vec"], ev_b[mech]["f05_vec"])


def checkpoint(name, art_dir, payload: dict, model=None):
    d = os.path.join(art_dir, name)
    os.makedirs(d, exist_ok=True)
    dump(payload, os.path.join(d, "summary.json"))
    if model is not None:
        model.save(os.path.join(d, "model.json"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write reports/artifacts/log under this dir (smoke/audit runs)")
    ap.add_argument("--no-write-configs", action="store_true", help="do not overwrite configs/*_final/decision.yaml")
    args = ap.parse_args()
    setup_logging("logs/run_experiments.log")
    base, mcfg = load_cfg("base"), load_cfg("model")
    if args.out:
        base["paths"]["reports"], base["paths"]["artifacts"] = f"{args.out}/reports", f"{args.out}/artifacts"
        os.makedirs(base["paths"]["reports"], exist_ok=True)
    R, A = base["paths"]["reports"], base["paths"]["artifacts"]
    label = base["metric_label"]
    elog = Log(label)
    T0 = time.time()

    # ---------------- Phase 1-3: data, validation, retrieval
    ctx = Context().prepare(base, mcfg)
    checkpoint("checkpoint_01_data_validation", A, {"label": label, "schema": ctx.schema})
    rr = ctx.write_retrieval_reports(R)
    rec_val = rr["retrieval_report_val"]["union"]["recall"]
    checkpoint("checkpoint_02_retrieval", A, {"label": label, "retrieval": rr, "prepare_s": ctx.prepare_s})
    log.info("union recall (all / val): %s / %s", rr["retrieval_report"]["union"]["recall"], rec_val)

    # ---------------- E1 baseline: unweighted binary GBDT, no experimental features
    cfg = copy.deepcopy(mcfg)
    cfg["features"].update(fs=False, relative_entity=False, relative_record=False)
    cfg["entity_weighting"] = "none"
    t = time.time()
    m, s, cols = train_model(ctx, cfg)
    ev_base = reference_eval(ctx, s)
    base_s, base_cfg, base_ev = s, copy.deepcopy(cfg), ev_base
    elog.add(id="E1", hypothesis="baseline: 4 sparse channels + unweighted binary GBDT",
             config=f"{len(cols)} feats, weighting=none", candidate_recall_val=rec_val, **_row(ev_base),
             runtime_s=round(time.time() - t, 1), diagnosis=str(pair_diagnostics(ctx, s)),
             outcome="baseline (top1_t ref: %s)" % _f(ev_base, "top1_t"))
    checkpoint("checkpoint_03_baseline_model", A, {"label": label, "config": cfg, "eval": ev_base,
                                                   "importance": m.importance()}, m)
    best = dict(cfg=cfg, s=s, ev=ev_base, model=m, cols=cols, extra_w=None)

    # ---------------- E2 entity weighting A/B
    cfg2 = copy.deepcopy(best["cfg"])
    cfg2["entity_weighting"] = "inv_count"
    t = time.time()
    m2, s2, _ = train_model(ctx, cfg2)
    ev2 = reference_eval(ctx, s2)
    gain = _f(ev2) - _f(best["ev"])
    bt = _bt(best["ev"], ev2)
    keep = bt["p_improve"] >= P_KEEP
    elog.add(id="E2", hypothesis="1/candidate_count entity weighting equalises per-entity gradient mass",
             config="weighting=inv_count", candidate_recall_val=rec_val, **_row(ev2), runtime_s=round(time.time() - t, 1),
             diagnosis=f"dF05={gain:+.4f} vs E1, paired bootstrap {bt}; no-match FP {ev2[REF]['strata']['overall']['no_match_fp_rate']} vs "
                       f"{best['ev'][REF]['strata']['overall']['no_match_fp_rate']}",
             outcome="KEPT" if keep else "REJECTED (no meaningful gain)")
    if keep:
        best.update(cfg=cfg2, s=s2, ev=ev2, model=m2)

    # ---------------- E2r ranking objective (comparison point; calibrated on calib to reach a 0-1 scale)
    cfgr = copy.deepcopy(best["cfg"])
    cfgr["objective"] = "rank"
    t = time.time()
    mr, sr, _ = train_model(ctx, cfgr)
    evr = reference_eval(ctx, sr, calibrate=True)
    ev_best_cal = reference_eval(ctx, best["s"], calibrate=True)
    nm_r, nm_b = (evr[REF]["strata"]["overall"]["no_match_fp_rate"], ev_best_cal[REF]["strata"]["overall"]["no_match_fp_rate"])
    gain = _f(evr) - _f(ev_best_cal)
    bt = _bt(ev_best_cal, evr)
    keep = bt["p_improve"] >= P_KEEP and nm_r <= nm_b
    elog.add(id="E2r", hypothesis="rank:pairwise objective vs binary (both isotonic-calibrated for comparability)",
             config="objective=rank", candidate_recall_val=rec_val, **_row(evr), runtime_s=round(time.time() - t, 1),
             diagnosis=f"dF05={gain:+.4f} vs calibrated binary, paired bootstrap {bt}; no-match FP rank={nm_r} binary={nm_b}",
             outcome="KEPT" if keep else "REJECTED (%s)" % ("worse no-match FP" if nm_r > nm_b else "no meaningful gain"))
    if keep:
        best.update(cfg=cfgr, s=sr, ev=evr, model=mr)

    # ---------------- E3 one round of stratified hard-negative mining (train_fit only)
    t = time.time()
    tr = ctx.rows("train_fit")
    cols = feature_columns(ctx.F.columns, best["cfg"]["features"])
    e_tr = ctx.cand["e"].values[tr]
    assert set(np.unique(e_tr)) <= set(ctx.idx["train_fit"]), "hard-negative pool leaks non-train entities"
    from src.models.gbdt import entity_weights
    w_tr = entity_weights(e_tr, best["cfg"]["entity_weighting"])
    oof = oof_scores(ctx.F.loc[tr, cols], ctx.y[tr], w_tr, e_tr, best["cfg"]["xgb"], best["cfg"]["hard_negative"]["folds"])
    has_true = ctx.n_true[e_tr] > 0
    mult, hn_counts = mine(ctx.F[tr].reset_index(drop=True), ctx.y[tr], oof, e_tr, has_true, best["cfg"]["hard_negative"])
    mined_ents = set(np.unique(e_tr[mult > 1]))
    assert mined_ents <= set(ctx.idx["train_fit"]), "mined hard negatives must come from train entities only"
    m3, s3, _ = train_model(ctx, best["cfg"], extra_weight=mult)
    ev3 = reference_eval(ctx, s3)
    gain = _f(ev3) - _f(best["ev"])
    bt = _bt(best["ev"], ev3)
    keep = bt["p_improve"] >= P_KEEP
    elog.add(id="E3", hypothesis="one round of stratified hard-negative up-weighting reduces confusable FPs",
             config=f"upweight={best['cfg']['hard_negative']['upweight']} strata={hn_counts}", candidate_recall_val=rec_val,
             **_row(ev3), runtime_s=round(time.time() - t, 1), diagnosis=f"dF05={gain:+.4f}, paired bootstrap {bt}; mined entities ⊆ train_fit (asserted)",
             outcome="KEPT" if keep else "REJECTED (no meaningful gain)")
    if keep:
        best.update(s=s3, ev=ev3, model=m3, extra_w=mult)
        best["cfg"]["hard_negative"]["enabled"] = True
    checkpoint("checkpoint_05_hard_negative", A, {"label": label, "kept": keep, "counts": hn_counts, "eval": ev3}, m3)

    # ---------------- E4 calibration + decision oracle
    def oracle(scores, tag):
        cal = ctx.rows("calib")
        calib = Calibrator().fit(scores[cal], ctx.y[cal])
        frames = {"raw": Frame(ctx.cand["e"].values, ctx.cand["r"].values, scores, ctx.y),
                  "cal": Frame(ctx.cand["e"].values, ctx.cand["r"].values, calib.transform(scores), ctx.y)}
        t = time.time()
        tab, detail, winner = run_oracle(frames, ctx.idx["calib"], ctx.idx["val"], ctx.n_true, ctx.cand_count, ctx.attrs)
        low = low_count_simulation(frames["cal"], ctx.idx["calib"], ctx.idx["val"], ctx.n_true, ctx.cand_count)
        write_oracle(tab, detail, winner, R, label, tag, low)
        return tab, detail, winner, calib, round(time.time() - t, 1)

    tab, detail, winner, calib, rt = oracle(best["s"], "decision_oracle")
    w = tab[tab.mechanism == winner].iloc[0]
    elog.add(id="E4", hypothesis="decision-layer oracle: 7 mechanism families x raw/calibrated x record-exclusive",
             config=f"winner={winner}", candidate_recall_val=rec_val, val_macro_f05=w.val_macro_f05,
             val_precision=w.val_pair_precision, val_recall=w.val_pair_recall, val_no_match_fp=w.val_no_match_fp_rate,
             **{"val_multi_p/r": f"{w.val_multi_precision}/{w.val_multi_recall}"}, decision=f"{winner} {w.params}",
             runtime_s=rt, diagnosis=f"{len(tab)} variants; see reports/decision_oracle.md", outcome="SELECTED")
    checkpoint("checkpoint_04_decision_layer", A, {"label": label, "winner": winner, "detail": detail})
    checkpoint("checkpoint_06_calibration", A, {"label": label, "winner_uses_calibration": winner.startswith("cal:")})
    best.update(winner=winner, detail=detail, calib=calib)

    # ---------------- E5 feature ablations (A = current, B = + group), stratified
    abl = {}
    ref_ev = reference_eval(ctx, best["s"])
    for grp, hyp in (("fs", "Fellegi-Sunter heuristic weight adds signal beyond raw agreements"),
                     ("relative_entity", "entity-side candidate-relative features (rank/gap/percentile)"),
                     ("relative_record", "record-side relative features (how contested a record is)")):
        c = copy.deepcopy(best["cfg"])
        c["features"][grp] = True
        t = time.time()
        mb, sb, _ = train_model(ctx, c, extra_weight=best["extra_w"])
        evb = reference_eval(ctx, sb)
        gain = _f(evb) - _f(ref_ev)
        bt = _bt(ref_ev, evb)
        worst = stratum_regressions(ref_ev[REF]["strata"], evb[REF]["strata"])
        keep = bt["p_improve"] >= P_KEEP and not worst
        abl[grp] = {"gain": gain, "bootstrap": bt, "regressions": worst, "A": ref_ev[REF]["strata"], "B": evb[REF]["strata"]}
        elog.add(id=f"E5-{grp}", hypothesis=hyp, config=f"+{grp}", candidate_recall_val=rec_val, **_row(evb),
                 runtime_s=round(time.time() - t, 1),
                 diagnosis=f"dF05={gain:+.4f}, paired bootstrap {bt}; strata regressing >0.02: {worst or 'none'}",
                 outcome="KEPT" if keep else ("REJECTED (stratum regression)" if worst else "REJECTED (no meaningful gain)"))
        if keep:
            best["cfg"] = c
            best.update(s=sb, model=mb)
            ref_ev = evb
    write_ablation(abl, R, label)
    changed = any(best["cfg"]["features"][g] for g in ("fs", "relative_entity", "relative_record"))
    if changed:  # upstream scores changed materially -> re-run the oracle once
        tab, detail, winner, calib, rt = oracle(best["s"], "decision_oracle")
        w = tab[tab.mechanism == winner].iloc[0]
        elog.add(id="E4b", hypothesis="oracle re-run after feature-set change", config=f"winner={winner}",
                 candidate_recall_val=rec_val, val_macro_f05=w.val_macro_f05, val_precision=w.val_pair_precision,
                 val_recall=w.val_pair_recall, val_no_match_fp=w.val_no_match_fp_rate,
                 **{"val_multi_p/r": f"{w.val_multi_precision}/{w.val_multi_recall}"}, decision=f"{winner} {w.params}",
                 runtime_s=rt, diagnosis="re-run because features changed", outcome="SELECTED")
        best.update(winner=winner, detail=detail, calib=calib)

    # ---------------- FINAL: baseline vs best system on the untouched TEST split
    final = final_eval(ctx, best, base_s, base_ev, R, label)
    for k, v in final.items():
        if "overall" not in v:
            continue
        o = v["overall"]
        elog.add(id=f"TEST-{k}", hypothesis="holdout evaluation (touched once)", config=v["decision"],
                 candidate_recall_val=rr["retrieval_report"]["union"]["recall"], val_macro_f05=o["macro_f05"],
                 val_precision=o["pair_precision"], val_recall=o["pair_recall"], val_no_match_fp=o["no_match_fp_rate"],
                 **{"val_multi_p/r": f"{o['multi_precision']}/{o['multi_recall']}"}, decision=v["decision"],
                 outcome="reported (TEST split; columns labelled val_* hold TEST numbers for these rows)")

    # persist the selected system for predict/submit
    mech_name = best["winner"]
    variant, mech = mech_name.split(":")
    excl = mech.endswith("+excl")
    mech = mech.replace("+excl", "")
    dec = {"calibrated": variant == "cal", "mechanism": mech, "params": best["detail"][mech_name]["params"],
           "exclusive_mode": "auto" if excl else "off", "tie_break": "low", "unseen_margin": 0.0, "retune": True}
    system = {"label": label, "base_cfg": base, "model_cfg": best["cfg"], "decision": dec, "candgen": ctx.cg,
              "tidf": ctx.tidf, "fs": ctx.fs, "booster_features": best["model"].features, "calibrator": best["calib"],
              "source_levels": list(pd.factorize(ctx.recs["source"])[1])}
    os.makedirs(os.path.join(A, "checkpoint_07_final"), exist_ok=True)
    best["model"].save(os.path.join(A, "checkpoint_07_final", "model.json"))
    with open(os.path.join(A, "checkpoint_07_final", "system.pkl"), "wb") as f:
        pickle.dump(system, f)
    dump({"label": label, "decision": dec, "model_cfg": best["cfg"], "final_test": final,
          "importance": best["model"].importance(30)}, os.path.join(A, "checkpoint_07_final", "summary.json"))
    import yaml
    if not args.no_write_configs:
        with open("configs/decision.yaml", "w", encoding="utf-8") as f:
            f.write("# Written by run_experiments.py from the decision-layer oracle (params tuned on calib fold).\n")
            yaml.safe_dump(json.loads(json.dumps(dec, default=float)), f, sort_keys=False)
        with open("configs/model_final.yaml", "w", encoding="utf-8") as f:
            f.write("# Final model config selected by the staged experiments (see EXPERIMENT_LOG.md).\n")
            yaml.safe_dump(best["cfg"], f, sort_keys=False)
    elog.write(os.path.join(args.out, "EXPERIMENT_LOG.md") if args.out else "EXPERIMENT_LOG.md")
    log.info("total runtime %.1fs", time.time() - T0)


def stratum_regressions(A, B, tol=0.02, min_n=25):
    out = []
    for k, a in A.items():
        b = B.get(k)
        if k == "overall" or b is None or a["n"] < min_n:
            continue
        if b["macro_f05"] < a["macro_f05"] - tol:
            out.append(f"{k}: {a['macro_f05']}->{b['macro_f05']}")
    return out


def final_eval(ctx, best, base_s, base_ev, R, label):
    out = {}
    # baseline system: baseline scores + its reference decision (tuned on calib)
    fr = Frame(ctx.cand["e"].values, ctx.cand["r"].values, base_s, ctx.y)
    p = base_ev[REF]["params"]
    acc_b = fr.accept(REF, p)
    pe = per_entity(fr.df["e"].values, acc_b, ctx.y, ctx.idx["test"], ctx.n_true, ctx.cand_count).merge(ctx.attrs, on="e")
    out["baseline"] = {"decision": f"raw:{REF} {p}", **stratified(pe, ("country", "true_sources"))}
    pe_base_f05 = pe["f05"].values
    # best system
    variant, mech = best["winner"].split(":")
    s = best["calib"].transform(best["s"]) if variant == "cal" else best["s"]
    fr = Frame(ctx.cand["e"].values, ctx.cand["r"].values, s, ctx.y)
    excl = mech.endswith("+excl")
    params = best["detail"][best["winner"]]["params"]
    if mech.startswith("hybrid"):
        acc = hybrid_accept(ctx, fr, params)
    else:
        acc = fr.accept(mech.replace("+excl", ""), params, "all" if excl else "off")
    pe = per_entity(fr.df["e"].values, acc, ctx.y, ctx.idx["test"], ctx.n_true, ctx.cand_count).merge(ctx.attrs, on="e")
    out["best"] = {"decision": f"{best['winner']} {params}", **stratified(pe, ("country", "true_sources"))}
    out["best_vs_baseline_paired_bootstrap"] = paired_bootstrap(pe_base_f05, pe["f05"].values)
    # error analysis + score analysis on TEST (reported) using the final system
    t = params.get("t") if isinstance(params, dict) else None
    sa = score_analysis(fr.df["e"].values, s, ctx.y, acc, ctx.idx["test"], ctx.n_true, t)
    sa["cand_bucket"] = pd.cut(sa.cand_count, [-1, 1, 4, 20, 10 ** 9], labels=["0-1", "2-4", "5-20", "20+"]).astype(str)
    sa["error_class"] = classify(sa)
    sa = sa.merge(ctx.attrs, on="e", how="left")
    sa.insert(0, "entity_id", ctx.eid[sa.e.values])
    sa.insert(0, "label", label)  # every synthetic number carries the label, CSVs included
    sa.to_csv(os.path.join(R, "entity_score_analysis.csv"), index=False)
    sa[sa.error_class != "correct"].to_csv(os.path.join(R, "error_analysis.csv"), index=False)
    tp_test = ctx.truth_pairs_for("test")
    got = set(zip(ctx.cand["e"].values, ctx.cand["r"].values)) & tp_test
    tree = diagnostic_tree(sa, len(got) / max(1, len(tp_test)))
    write_error_md(sa, tree, R, label, ctx, fr, acc)
    return out


def hybrid_accept(ctx, fr, choice):
    bucket = pd.cut(pd.Series(ctx.cand_count), [-1, 1, 4, 20, 10 ** 9], labels=["0-1", "2-4", "5-20", "20+"]).astype(str).values
    rb = bucket[fr.df["e"].values]
    acc = np.zeros(len(fr.df), bool)
    for b, c in choice.items():
        acc |= fr.accept(c["mechanism"], c["params"]) & (rb == b)
    return acc


def write_oracle(tab, detail, winner, R, label, name, low=None):
    tab.assign(label=label).to_csv(os.path.join(R, f"{name}.csv"), index=False)
    dump({"label": label, "winner": winner, "detail": detail}, os.path.join(R, f"{name}.json"))
    cols = ["mechanism", "calib_f05", "val_macro_f05", "val_pair_precision", "val_pair_recall", "val_no_match_fp_rate",
            "val_multi_precision", "val_multi_recall", "val_fp_per_entity", "val_fn_per_entity", "val_accepted_per_entity"]
    strata = [c for c in tab.columns if c.startswith("f05[cand_bucket") or c.startswith("f05[gt_type")]
    lines = [f"# Decision-layer oracle\n\n**{label}**\n",
             "Frozen candidates + scores; only the mechanism varies. Parameters tuned on the calib fold "
             "(carved from train); metrics on VAL. `cal:` = isotonic-calibrated scores (fit on calib). "
             "`+excl` = record exclusivity over ALL candidates (Source 1 is deduplicated; shipped as exclusive_mode=auto, enabled only if train GT links no record to >1 entity).\n",
             f"**Selected: `{winner}`**. Rule: take the best VAL macro F0.5 that does not collapse (>0.05 below the "
             "best) in the 0-1 / 2-4 candidate buckets or the no-match stratum; it replaces the default `cal:multi` "
             "only if a paired bootstrap over VAL entities gives P(improve) >= 0.9. "
             f"Selection detail: {detail['_selection']}\n", "## Overall", "",
             "| " + " | ".join(cols + ["params", "collapses", "paired bootstrap vs default"]) + " |",
             "|" + "---|" * (len(cols) + 3)]
    for _, r in tab.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + f" | {r['params']} | {r['collapses'] or ''} | "
                     f"{r['bootstrap_vs_default']} |")
    lines += ["", "## Stratified macro F0.5 (VAL)", "", "| mechanism | " + " | ".join(strata) + " |", "|" + "---|" * (len(strata) + 1)]
    for _, r in tab.iterrows():
        lines.append(f"| {r['mechanism']} | " + " | ".join(str(r[c]) for c in strata) + " |")
    lines += ["", "## Country / true-source strata for the selected mechanism", "",
              "| stratum | n | macro F0.5 | no-match FP | FP/entity | FN/entity |", "|---|---|---|---|---|---|"]
    for k, v in detail[winner]["val"].items():
        lines.append(f"| {k} | {v['n']} | {v['macro_f05']} | {v['no_match_fp_rate']} | {v['fp_per_entity']} | {v['fn_per_entity']} |")
    if low is not None:
        low.assign(label=label).to_csv(os.path.join(R, f"{name}_low_count_sim.csv"), index=False)
        lines += ["", "## Low-candidate-count simulation (calibrated scores; lists truncated to top-n)", "",
                  "The synthetic retriever never yields <5 candidates, so the natural 1 and 2-4 buckets are empty. "
                  "Here every entity keeps only its top-n candidates; parameters tuned on calib, scored on VAL.", "",
                  "| n | mechanism | val F0.5 | no-match F0.5 | single F0.5 | no-match FP | params |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in low.iterrows():
            lines.append(f"| {r.n_candidates} | {r.mechanism} | {r.val_macro_f05} | {r.val_f05_no_match} | "
                         f"{r.val_f05_single} | {r.val_no_match_fp} | {r.params} |")
    with open(os.path.join(R, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_ablation(abl, R, label):
    lines = [f"# Candidate-relative / FS feature ablation\n\n**{label}**\n",
             "A = current best feature set; B = A + group. Reference decision (multi, tuned on calib). VAL split.\n"]
    for g, d in abl.items():
        lines += [f"## +{g}: dF05 = {d['gain']:+.4f}; paired bootstrap {d['bootstrap']}; stratum regressions > 0.02: {d['regressions'] or 'none'}", "",
                  "| stratum | n | A F0.5 | B F0.5 | A no-match FP | B no-match FP |", "|---|---|---|---|---|---|"]
        for k, a in d["A"].items():
            b = d["B"].get(k, {})
            lines.append(f"| {k} | {a['n']} | {a['macro_f05']} | {b.get('macro_f05')} | {a['no_match_fp_rate']} | {b.get('no_match_fp_rate')} |")
        lines.append("")
    with open(os.path.join(R, "feature_ablation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_error_md(sa, tree, R, label, ctx, fr, acc):
    lines = [f"# Error analysis (TEST split, final system)\n\n**{label}**\n", "## Diagnostic tree", *tree, "",
             "## Error class by stratum", ""]
    for col in ("cand_bucket", "country", "true_sources"):
        lines += ["```", pd.crosstab(sa["error_class"], sa[col]).to_string(), "```", ""]
    # manual-inspection samples
    d = fr.df.assign(y=ctx.y, acc=acc)
    d = d[d.e.isin(set(ctx.idx["test"]))]
    names_e, names_r = ctx.ents["name"].values, ctx.recs["name"].values
    addr_e, addr_r = ctx.E["addr_fold"].values, ctx.R["addr_fold"].values

    def show(sub, title, n=8):
        lines.extend([f"### {title}", "", "| entity | record | score | label | S1 name / addr | record name / addr |", "|---|---|---|---|---|---|"])
        for _, r in sub.head(n).iterrows():
            lines.append(f"| {ctx.eid[r.e]} | {ctx.recs.record_id.values[r.r]} | {r.s:.3f} | {r.y} | "
                         f"{names_e[r.e]} / {addr_e[r.e]} | {names_r[r.r]} / {addr_r[r.r]} |")
        lines.append("")
    show(d[d.acc].sort_values("s", ascending=False), "Highest-confidence accepted matches")
    show(d[d.acc].sort_values("s"), "Lowest-confidence accepted matches")
    show(d[d.acc & (d.y == 0)].sort_values("s", ascending=False), "False positives (accepted, label 0)")
    show(d[~d.acc & (d.y == 1)].sort_values("s", ascending=False), "False negatives among retrieved (rejected, label 1)")
    nm = sa[(sa.n_true == 0)].sort_values("best_score", ascending=False)
    show(d[d.e.isin(set(nm.e.head(8)))].sort_values("s", ascending=False).groupby("e").head(1), "No-match entities: best candidate")
    miss = [(e, r) for e, r in ctx.truth_pairs_for("test") if (e, r) not in set(zip(d.e, d.r))]
    lines += ["### Retrieval misses (true pair never retrieved)", "", "| entity | S1 name / addr | record name / addr |", "|---|---|---|"]
    for e, r in miss[:12]:
        lines.append(f"| {ctx.eid[e]} | {names_e[e]} / {addr_e[e]} | {names_r[r]} / {addr_r[r]} |")
    with open(os.path.join(R, "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
