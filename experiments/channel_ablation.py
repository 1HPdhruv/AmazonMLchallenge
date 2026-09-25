"""E6: end-to-end A/B of retrieval channels (Section 8 marginal-recall rule), final model config +
calibrated multi decision tuned on calib, scored on VAL. SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE."""
from __future__ import annotations

import copy
import json
import logging
import os
import sys

sys.path.insert(0, os.getcwd())
from src.evaluation.entity_metrics import paired_bootstrap  # noqa: E402
from src.pipeline.core import Context, load_cfg, reference_eval, train_model  # noqa: E402

logging.disable(logging.INFO)


def run(base, mcfg):
    ctx = Context().prepare(base, mcfg)
    _, s, _ = train_model(ctx, mcfg)
    ev = reference_eval(ctx, s, mechs=("multi",), calibrate=True)["multi"]
    rec = ctx.truth_pairs_for("val")
    got = len(set(zip(ctx.cand.e, ctx.cand.r)) & rec) / len(rec)
    return ev, round(got, 4), len(ctx.cand)


def main():
    base, mcfg = load_cfg("base"), load_cfg("model_final")
    A = run(base, mcfg)
    rows = {"all_channels": A}
    out = {"label": base["metric_label"], "variants": {}}
    for ch in ("word", "exact"):
        b = copy.deepcopy(base)
        b["retrieval"]["channels"][ch] = False
        b["paths"]["reports"] = "experiments/channel_ablation"
        rows[f"no_{ch}"] = run(b, mcfg)
    for k, (ev, rec, n) in rows.items():
        o = ev["strata"]["overall"]
        bt = paired_bootstrap(A[0]["f05_vec"], ev["f05_vec"]) if k != "all_channels" else None
        out["variants"][k] = {"val_union_recall": rec, "pairs": n, "val_macro_f05": o["macro_f05"],
                              "no_match_fp": o["no_match_fp_rate"], "params": ev["params"], "bootstrap_vs_all": bt}
        print(k, out["variants"][k], flush=True)
    json.dump(out, open("reports/channel_ablation.json", "w", encoding="utf-8"), indent=2, default=float, ensure_ascii=False)
    append_log(out)


def append_log(out):
    """Adds/replaces the E6 rows in EXPERIMENT_LOG.md (run after run_experiments)."""
    path = "EXPERIMENT_LOG.md"
    lines = [l for l in open(path, encoding="utf-8").read().splitlines() if not l.startswith("| E6")]
    a = out["variants"]["all_channels"]
    for k in ("no_word", "no_exact"):
        v = out["variants"][k]
        bt = v["bootstrap_vs_all"]
        keep_removal = bt["p_improve"] >= 0.9
        lines.append(f"| E6-{k} | removing the {k[3:]} channel loses no entity-level quality (Section 8 marginal-recall rule) | "
                     f"retrieval.channels.{k[3:]}=false; final model cfg; cal:multi tuned on calib | {v['val_union_recall']} "
                     f"(all: {a['val_union_recall']}) | {v['val_macro_f05']} (all: {a['val_macro_f05']}) |  |  | "
                     f"{v['no_match_fp']} (all: {a['no_match_fp']}) |  | cal:multi {v['params']} |  | "
                     f"pairs {v['pairs']} vs {a['pairs']}; paired bootstrap {bt} | "
                     f"{'channel REMOVED' if keep_removal else 'channel KEPT (removal not justified end-to-end)'} |")
    open(path, "w", encoding="utf-8").write(chr(10).join(lines) + chr(10))


if __name__ == "__main__":
    main()
