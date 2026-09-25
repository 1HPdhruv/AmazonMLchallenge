# Final engineering report: Business Entity Resolution

> **Status note (2026-09-25 audit):** this report describes the **v1** synthetic run, made before the
> competition-schema audit. v1's synthetic data had structured addresses and phone numbers, which the real
> data does not have. The audit removed both, switched to the official TSV layout and outputs, and preserved
> v1 in `artifacts/synthetic_best_v1/`. All conclusions below are provisional and must be redone on real data.

> **SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE.** Every metric in this report comes from a
> synthetic benchmark I generated (`tests/synthetic/generate_data.py`). None of them predicts
> competition or leaderboard performance. The synthetic generator's design choices (noise rates,
> chain structure, address formats) drive every number below.

## A. What was built
A complete, config-driven entity-resolution pipeline, tested end to end:
- a `CompetitionDataAdapter` (canonical schema, mapping from raw columns to canonical fields, free-text address parsing);
- schema/data validation that fails loudly on critical problems;
- additive normalized representations;
- four sparse retrieval channels whose weights are fit on train only, plus a union and a single retrieval-report generator;
- pair features;
- an XGBoost pair classifier with an entity-weighting option, a ranking comparison point and one round of stratified hard-negative mining;
- isotonic calibration on a calib fold carved from train;
- a decision-layer oracle (7 mechanism families × raw/calibrated × record-exclusive, a per-bucket hybrid and a low-candidate-count simulation);
- a six-class error taxonomy with a diagnostic tree;
- train, predict and submit entry points with a submission validator;
- seven phase checkpoints, 20 tests, and runtime-scaling and determinism checks.

## B. Data mode
**DATASET UNAVAILABLE MODE.** The repository was empty. No competition files, sample submission or validator were found in the project directory, `D:\`, Downloads, Desktop or Documents.

## C. Dataset sizes
Real sizes: **not observed**. Synthetic benchmark: 6,000 Source-1 entities; 12,090 records (source2 6,736, source3 5,354); 6,021 true pairs. Match types: 2,135 no-match, 2,629 single, 1,236 multi (max 4). Splits by entity: train_fit 2,700 / calib 900 / val 1,200 / test 1,200.

## D. Observed schema
Real schema: **not observed**. Synthetic schema (chosen to exercise the adapter):
- S1 has structured name, address, city, region, postal, country and phone.
- S2 has structured fields plus phone, under different column names.
- S3 has a name and one free-text `full_address`, parsed by the adapter.

The phone identifier features are built only because both sides have phone. That condition is checked in code.

## E. Retrieval (all 6,000 entities; `reports/retrieval_report.md`, VAL-only in `retrieval_report_val.md`)
| channel | recall | unique true pairs (only this channel) | marginal recall | P50 | P95 | P99 | max | runtime s |
|---|---|---|---|---|---|---|---|---|
| exact key | 0.4288 | 0 | 0.0000 | 0 | 18 | 24 | 29 | 0.009 |
| char TF-IDF | 0.8524 | 163 | 0.0271 | 20 | 20 | 20 | 20 | 1.28 |
| word TF-IDF | 0.6768 | 8 | 0.0013 | 3 | 20 | 20 | 20 | 0.53 |
| address keys | 0.8525 | 722 | 0.1199 | 20 | 20 | 20 | 20 | 0.23 |
| **union** | **0.9806** (VAL 0.9809) | – | – | 38 | 51 | 58 | 59 | 5.2 total incl. encoding |

The retrieval sweep on **train entities only** (`reports/retrieval_sweep.md`) raised the char/word cosine floor from 0.2 to 0.4: same train recall (0.9798), 26% fewer candidates. E6 (end-to-end channel ablation): removing the word channel left VAL F0.5 unchanged (0.952, P(improve)=0.51) but raised no-match FP from 0.026 to 0.037. Removing exact gave P(improve)=0.19. **Both kept.** The word channel fails the retrieval-only volume heuristic (+0.13% recall for +32% P99), which is flagged for re-check on real data.

## F. Model experiments (VAL, reference decision `multi` tuned on calib; keep only if paired-bootstrap P(improve) ≥ 0.9)
| ID | change | VAL macro F0.5 | no-match FP | P(improve) | outcome |
|---|---|---|---|---|---|
| E1 | baseline: unweighted binary GBDT, 52 features | 0.9487 | 0.0256 | – | baseline |
| E2 | 1/candidate_count entity weighting | 0.9503 | 0.0209 | 0.758 | rejected (not significant) |
| E2r | rank:pairwise (vs calibrated binary) | 0.9371 vs 0.952 | 0.0395 vs 0.0256 | 0.000 | rejected (worse, higher no-match FP) |
| E3 | 1 round stratified hard negatives (257 rows / 249 train entities; all 5 strata populated) | 0.9516 | 0.0419 | 0.803 | rejected (not significant) |
| E5-fs | + Fellegi–Sunter weight | 0.9476 | 0.0186 | 0.352 | rejected (5-20 bucket 0.987→0.961) |
| E5-rel-entity | + candidate-relative (entity side) | 0.9492 | 0.0535 | 0.565 | rejected (no-match stratum 0.974→0.947) |
| E5-rel-record | + candidate-relative (record side) | 0.9516 | 0.0395 | 0.780 | rejected (not significant) |

Pair-level AUC is 0.9998 (diagnostic only). Entity-level errors come from recall and decisions, not from ranking.

## G. Decision-layer oracle (`reports/decision_oracle.md`; params tuned on calib, scored on VAL)
- **Selected: `cal:multi` (isotonic, t=0.40, δ=0.20).** VAL F0.5 0.952, pair precision 0.980, recall 0.893, no-match FP 0.026, multi-match P/R 0.988/0.855.
- The top raw VAL variant was `raw:multi+excl` (0.9535). It did not pass the significance gate (P(improve)=0.66) and relies on an unverified assumption that each record belongs to at most one entity, so it was not adopted.
- threshold / multi_t2 / GMM on calibrated scores: 0.950–0.9505. `multi_t2` (a targeted attempt at multi-match under-selection) converged to the plain threshold solution, so no gain.
- top-1 families: ~0.91–0.915, because recall on multi-match entities collapses (pair recall 0.61). Pure top-1: 0.71, and flagged as collapsing on no-match.
- Strata for the selected mechanism: no-match 0.974, single 0.940, multi 0.938. Countries range 0.935 (JP) to 0.956 (BR). The true-source strata are the weak spot: source3-only 0.909, source2+source3 0.935.
- **Candidate-count strata:** the synthetic retriever never produces fewer than 5 candidates (1,160/1,200 VAL entities have 20+), so the natural 1 and 2–4 buckets are empty. In the controlled top-n truncation simulation, GMM does not collapse but trails `multi` at n=2–4 (0.9395 / 0.945 / 0.9486 vs 0.9421 / 0.9505 / 0.952), and its single-match F0.5 drops at n≥3. It stays benchmark-only.

## H. Baseline vs best validated system (TEST split, touched once per system)
| system | macro F0.5 | pair P | pair R | no-match FP | multi P/R |
|---|---|---|---|---|---|
| baseline: E1 model + raw `multi` (t=0.965, δ=0.05) | 0.9394 | 0.9751 | 0.9124 | 0.0175 | 0.991/0.916 |
| **final: same model + `cal:multi` (t=0.40, δ=0.20)** | **0.9443** | 0.9858 | 0.8948 | 0.0175 | 0.994/0.880 |

Paired bootstrap (final − baseline): +0.0049, 90% CI [+0.0017, +0.0082], P(improve)=0.998. The whole gain comes from the calibrated decision layer. No model-side change survived. Re-reading `submissions/submission.csv` from disk and scoring it gives exactly 0.9443 (end-to-end consistency).

## I. Final architecture (enabled)
Adapter → validation → normalized/folded/structured representations → retrieval {exact, char TF-IDF, word TF-IDF, address keys} → union → features {lexical, IDF, address components, phone, missingness, retrieval} → XGBoost binary (unweighted, `scale_pos_weight`) → isotonic calibration → `multi` decision.

## J. Tested and not retained
- Entity weighting (E2), ranking objective (E2r), hard negatives (E3), FS weight and both candidate-relative groups (E5): reasons in F.
- Record-exclusive post-filter, GMM, top-1 variants, per-bucket hybrid and `multi_t2`: reasons in G.
- Channel removals (E6): reasons in E.
- **Dense retrieval: NO-GO.** Union recall is 0.981, above the 0.95 line. Misses are acronyms (non-lexical by design), records with a missing name, and chain saturation, where identically named branches fill the top-k (debugged example: cosine 0.914 ranked 27th with k=20).
- **Cross-encoder: DEFERRED, NOT BUILT.** Criteria 1–2 are met on synthetic data (24% of test errors fall in the [0.2, 0.8] score band, mostly single-match `decision_miss`). Criterion 3 (size/license) cannot be verified without the rules, and it needs a model download that was not approved.
- **LLM matching:** excluded outright, per the directive.

## K. Final configuration
- channels: exact (≤30 postings), char and word TF-IDF (k=20, cosine ≥0.4), address keys (k=20, IDF floor 3.0, ≤50 postings)
- XGBoost: 400 trees, depth 6, η=0.05, subsample/colsample 0.8, hist, seed 7, `scale_pos_weight` = neg/pos
- weighting none; hard negatives off; FS and relative features off
- isotonic calibration on calib
- decision `multi` t=0.40 δ=0.20; record-exclusive off

These are recorded in `configs/model_final.yaml` and `configs/decision.yaml`.

## L. Submission
**No official competition submission was generated because the actual competition dataset is unavailable.** Submission mechanics were validated on the synthetic holdout with a placeholder format (`submissions/submission.csv`: one row per Source-1 entity, space-separated record IDs, empty string for no match). All 8 validator checks pass (`submissions/submission_validation_report.md`).

## M. Reproduction
```bash
pip install -r requirements.txt
python -m src.pipeline.run_experiments && python experiments/channel_ablation.py && python experiments/phase9_gate.py
python -m src.pipeline.train && python -m src.pipeline.predict --entities test && python -m src.pipeline.submit
python -m pytest -q
```

## N. Remaining risks (evidence-based)
1. **Nothing here has been checked against real data.** The schema mapping, address parser, legal-form and abbreviation lists, and every tuned parameter are fit to synthetic conventions. Expect to re-tune retrieval floors and decision parameters on real data, and keep the selection gates unchanged.
2. **The weakest stratum is the free-text-address source.** Source3-only entities score 0.909 VAL / 0.887 TEST, versus 0.964 VAL / 0.936 TEST for source2-only. The heuristic `parse_full_address` is the likely weak point for real, unfamiliar formats.
3. **Multi-match under-selection is the largest error class** (66 of 137 test failures). Precision-leaning F0.5 plus calibration pushes the decision toward rejecting weaker true duplicates, and neither `multi_t2` nor δ ≤ 0.2 recovered them without adding false positives.
4. **The low-candidate-count strata were never observed.** The GMM and low-count guard results come only from a truncation simulation.
5. **Chain saturation.** Identical-name branches can push a true record out of the top-k. Records without an address can't be attributed to a branch anyway.
6. **Unused training labels.** The GBDT is trained on train_fit only (75% of train labels) so that calibration and thresholds stay out-of-sample. Retraining on train_fit+calib would invalidate the tuned decision parameters unless they were re-tuned.
7. **Throughput.** The per-pair feature loops run at roughly 100 s per million candidate pairs (10,000 entities: 385k pairs, 46 s prepare). A much larger real dataset needs chunking or parallelism.
8. **Environment.** On the build machine numpy/scipy/scikit-learn/PyYAML resolve through a global `PYTHONPATH=D:\Quantum`. Pin `requirements.txt` in a clean environment.
9. **The submission format is a placeholder** and has not been checked against an official validator.
