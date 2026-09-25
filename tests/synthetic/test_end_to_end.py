"""End-to-end on the synthetic tree (official layout): train on train files -> save -> load ->
infer on test files (incl. unseen country France) -> output/*.tsv -> local checks -> official-
validator gate -> re-score from file. SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE."""
import pickle

import numpy as np

from src.data.adapters import make_adapter
from src.pipeline.core import load_cfg
from src.pipeline.predict import run_inference, score
from src.pipeline.submit import CAND_COLS, MATCH_COLS, local_checks, official_gate, read_output, score_matching
from src.pipeline.train import build_system


def test_end_to_end(small_cfg, tmp_path):
    dcfg = {"calibrated": True, "mechanism": "multi", "params": {"t": 0.5, "delta": 0.2}, "record_exclusive": False}
    system, model, ctx, s = build_system(small_cfg, load_cfg("model"), dcfg)
    sys_path, model_path = tmp_path / "system.pkl", tmp_path / "model.json"
    model.save(str(model_path))
    with open(sys_path, "wb") as f:
        pickle.dump(system, f)
    with open(sys_path, "rb") as f:
        system = pickle.load(f)

    # reloaded system reproduces in-memory train scores exactly
    tr_ents, tr_recs, _ = make_adapter(small_cfg).load("train")
    assert np.allclose(score(system, tr_ents, tr_recs, str(model_path))["s_raw"].values, s, atol=1e-6)

    ad = make_adapter(small_cfg)
    ents, recs, labels = ad.load("test")
    assert labels is None                                   # test labels are never loaded by the pipeline
    assert "France" in set(ents["country"])                 # unseen country present at inference
    out = small_cfg["output"]["dir"]
    paths, stats = run_inference(system, ents, recs, str(model_path), out)
    assert local_checks(out, ents["entity_id"].tolist(), set(recs["record_id"])) == []
    h, mrows = read_output(paths["matching_results.tsv"], MATCH_COLS)
    h2, crows = read_output(paths["candidate_pairs.tsv"], CAND_COLS)
    assert h == MATCH_COLS and h2 == CAND_COLS
    assert [r[0] for r in mrows] == ents["entity_id"].tolist() == [r[0] for r in crows]  # every S1 incl. France
    cand = {e: set(v.split(",")) if v else set() for e, v, _ in crows}
    assert all((set(v.split(",")) if v else set()) <= cand[e] for e, v, _ in mrows)
    assert stats["candidate_pairs"] == sum(len(v) for v in cand.values())
    gate = official_gate(small_cfg, out)                    # synthetic tree has no official validator
    assert gate["ran"] is False
    lab = ad.hidden_test_labels(set(ents.entity_id), set(recs.record_id))
    assert 0.0 <= score_matching(paths["matching_results.tsv"], lab, ents["entity_id"].tolist()) <= 1.0
