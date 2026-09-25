"""Runtime / memory scaling on synthetic data (engineering test only — never a performance estimate).
SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE."""
from __future__ import annotations

import copy
import gc
import json
import logging
import os
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd())
import psutil  # noqa: E402

from src.pipeline.core import Context, load_cfg, train_model  # noqa: E402
from tests.synthetic.generate_data import generate  # noqa: E402

logging.disable(logging.INFO)


def main(sizes=(10, 100, 1000, 10000)):
    proc = psutil.Process()
    rows = []
    for n in sizes:
        d = tempfile.mkdtemp(prefix=f"scale{n}_")
        generate(n, 5, d)
        base = copy.deepcopy(load_cfg("base"))
        base["data"]["root"] = d
        base["paths"]["reports"] = d
        gc.collect()
        rss0 = proc.memory_info().rss
        t = time.time()
        try:
            ctx = Context().prepare(base, load_cfg("model"))
            t_prep = time.time() - t
            t = time.time()
            train_model(ctx, load_cfg("model"))
            t_train = time.time() - t
            row = dict(n_entities=n, n_records=len(ctx.recs), n_pairs=len(ctx.cand), prepare_s=round(t_prep, 2),
                       train_predict_s=round(t_train, 2), rss_delta_mb=round((proc.memory_info().rss - rss0) / 2 ** 20, 1),
                       status="ok")
        except Exception as ex:  # tiny sizes can legitimately lack positives in train_fit
            row = dict(n_entities=n, status=f"failed: {type(ex).__name__}: {ex}"[:200])
        rows.append(row)
        print(row, flush=True)
        del_ctx = None  # noqa: F841
        gc.collect()
    out = {"label": "SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE (engineering scaling only)", "rows": rows}
    json.dump(out, open("reports/scaling.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    lines = ["# Scaling test\n", f"**{out['label']}**\n", "| entities | records | candidate pairs | prepare s | train+score s | RSS delta MB | status |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['n_entities']} | {r.get('n_records', '')} | {r.get('n_pairs', '')} | {r.get('prepare_s', '')} | "
                     f"{r.get('train_predict_s', '')} | {r.get('rss_delta_mb', '')} | {r['status']} |")
    open("reports/scaling.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
