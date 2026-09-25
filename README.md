# Amazon ML Challenge 2026: Business Entity Resolution

> **Mode: DATASET UNAVAILABLE MODE.** The official `student_resource` bundle (data + validator) was not
> present when this was built. Every number in this repository comes from a synthetic
> benchmark and is labelled **SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE**. None of
> these numbers predicts leaderboard performance.
> **No official competition submission was generated because the actual competition dataset is unavailable.**

Full results and decisions are in [reports/FINAL_ENGINEERING_REPORT.md](reports/FINAL_ENGINEERING_REPORT.md).
Configuration and data-drop-in steps are in [CONFIG.md](CONFIG.md). Every experiment is in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

The official data/output contract is in [configs/competition_facts.md](configs/competition_facts.md).

## Pipeline
```
dataset/{train,test}/*_source{1,2,3}.tsv ──► CompetitionDataAdapter (official TSV layout; 4 fields only)
  ──► schema validation (fails loudly) ──► representations (normalized, accent-folded, name key,
      address components parsed from business_address)
  ──► retrieval, stats fit on train only: exact key | char TF-IDF | word TF-IDF | rarity-weighted address keys
  ──► union + dedup ═══► output/candidate_pairs.tsv (exactly the pairs the model scores)
  ──► pair features (lexical, IDF, address components, missingness, retrieval)
  ──► XGBoost binary pair classifier ──► isotonic calibration (calib fold)
  ──► decision (provisional: calibrated multi-match t=0.4, δ=0.2)
  ──► output/matching_results.tsv  (asserted ⊆ candidates; local checks; official validator = hard gate)
```

## Reproduce (clean environment; does not depend on any global PYTHONPATH)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q                       # 38 tests
.venv/Scripts/python -m src.pipeline.train              # fit on dataset/train -> artifacts/final/
.venv/Scripts/python -m src.pipeline.predict            # dataset/test -> output/*.tsv + gate
```
Synthetic mode generates `data/synthetic/` (official layout) on first use. Full experiment matrix:
`python -m src.pipeline.run_experiments` (~4 min). It overwrites `configs/model_final.yaml` and
`configs/decision.yaml`; add `--out <dir> --no-write-configs` for a smoke run.
Optional: `python experiments/retrieval_sweep.py` (retrieval k / floor sweep on train entities) and
`python experiments/scaling.py` (runtime scaling from 10 to 10,000 entities).

The run is deterministic: candidates, scores and output files are bit-identical across processes with different `PYTHONHASHSEED` values (`tests/integration/test_determinism.py`).

## Layout
`src/data` adapter, validation, splits · `src/preprocessing` normalize, address, representations ·
`src/retrieval` TF-IDF, exact/address channels, union + retrieval report · `src/features` pair features, FS weight ·
`src/models` GBDT, hard negatives · `src/decision` mechanisms (one dispatcher), calibration, oracle ·
`src/evaluation` entity-level F0.5, stratification, error taxonomy · `src/pipeline` core, experiments, train, predict, submit ·
`tests/` unit, integration, synthetic generator + end-to-end · `reports/` all generated reports ·
`artifacts/checkpoint_0*` phase checkpoints.
