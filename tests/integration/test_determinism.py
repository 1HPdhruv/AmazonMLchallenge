"""Determinism across processes: two runs under different PYTHONHASHSEED values must produce
bit-identical candidate sets, scores and output files."""
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent("""
    import hashlib, json, logging, pickle, sys
    import numpy as np
    logging.disable(logging.INFO)
    from src.pipeline.core import load_cfg
    from src.pipeline.train import build_system
    from src.pipeline.predict import run_inference
    from src.data.adapters import make_adapter
    base = json.loads(sys.argv[1])
    dcfg = {"calibrated": True, "mechanism": "multi", "params": {"t": 0.5, "delta": 0.2}, "record_exclusive": False}
    system, model, ctx, s = build_system(base, load_cfg("model"), dcfg)
    model.save(base["output"]["dir"] + "/m.json")
    ents, recs, _ = make_adapter(base).load("test")
    paths, _ = run_inference(system, ents, recs, base["output"]["dir"] + "/m.json", base["output"]["dir"])
    h = hashlib.sha256()
    h.update(ctx.cand[["e", "r"]].values.tobytes()); h.update(np.round(s, 7).tobytes())
    for p in sorted(paths.values()): h.update(open(p, "rb").read())
    print(h.hexdigest())
""")


def test_cross_process_determinism(small_cfg, tmp_path):
    import json
    digests = []
    for seed in ("1", "12345"):
        cfg = json.loads(json.dumps(small_cfg))
        cfg["output"]["dir"] = str(tmp_path / f"o{seed}")
        os.makedirs(cfg["output"]["dir"])
        env = dict(os.environ, PYTHONHASHSEED=seed)
        env["PYTHONPATH"] = os.pathsep.join([os.getcwd()] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        res = subprocess.run([sys.executable, "-c", SCRIPT, json.dumps(cfg)], env=env, capture_output=True, text=True,
                             cwd=os.getcwd())
        assert res.returncode == 0, res.stderr[-3000:]
        digests.append(res.stdout.strip().splitlines()[-1])
    assert digests[0] == digests[1]
