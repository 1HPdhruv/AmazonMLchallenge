# Configuration

All parameters live in `configs/`; nothing experiment-relevant is hard-coded.

| file | what it holds | written by |
|---|---|---|
| `competition_facts.md` | the verified official data / output / rules contract | hand (from the official PDFs) |
| `base.yaml` | mode, metric label, seed, `data.root` (official layout), validator path, split fractions, retrieval channels / k / floors, output dir | hand |
| `model.yaml` | default GBDT params, objective, entity weighting, feature-group flags, hard-negative settings | hand |
| `model_final.yaml` | model config chosen by the staged experiments (**provisional: synthetic**) | `run_experiments.py` |
| `decision.yaml` | decision mechanism + parameters chosen by the oracle, tuned on the calib fold (**provisional: synthetic**) | `run_experiments.py` |

## Switching to real data (REAL DATA MODE)
The adapter reads the official layout directly, so no column mapping is needed:
`<root>/dataset/{train,test}/{split}_source{1,2,3}.tsv` plus `<root>/dataset/train/train_ground_truth.tsv`.
Only `entity_id, business_name, business_address, country` are used. Address components are parsed from
`business_address`, and country is an open set.
1. Unzip `student_resource` somewhere; never modify it.
2. In `base.yaml` set `mode: real`, `metric_label: ""`, `data.root: <path to student_resource>`.
3. `python -m src.pipeline.run_experiments`: re-selects retrieval / model / decision on REAL train labels
   (entity-level splits of the train files) and writes `model_final.yaml` + `decision.yaml`.
4. `python -m src.pipeline.train`, then `python -m src.pipeline.predict` writes `output/matching_results.tsv` and
   `output/candidate_pairs.tsv` for every test S1 entity. It aborts if matches ⊄ candidates, if any local rule
   fails, or if `utils/validate_submission.py` is missing or fails (**hard gate**).
5. Log every upload in `submissions/leaderboard_log.md`.

## Train-only statistics (hard rule, enforced in code)
| statistic | fit on | enforcement |
|---|---|---|
| char / word TF-IDF IDF (hashed; unseen token → max IDF) | `train_fit` Source-1 entities | `CandidateGenerator.fit(E[train_fit])`; test `test_train_only_statistics` asserts `n_fit_docs == len(train_fit)` |
| address-key IDF (rarity floor) | `train_fit` entities | same |
| token-IDF / rare-token threshold for features | `train_fit` names | `TokenIdf(..., E[train_fit])` |
| Fellegi–Sunter m/u | `train_fit` labelled candidate pairs | `fs.fit(F[train_fit rows])` |
| GBDT | `train_fit` rows | `train_model` |
| hard-negative selection | `train_fit` OOF scores | `assert mined ⊆ train_fit` |
| isotonic calibration | `calib` fold (carved from train) | `Calibrator().fit(s[calib])` |
| thresholds / margin / δ | `calib` fold | `tournament.tune(..., calib_ents)` |
| mechanism / component selection | `val` | paired bootstrap, P(improve) ≥ 0.9 |
| final reported number | `test` (touched once per reported system) | `final_eval` |

The retrieval **index** contains all records (as at inference time); only weight fitting is train-restricted.

## Selection rules
- Model-stage A/B (E2, E2r, E3, E5): keep a change only if the paired bootstrap over VAL entities gives P(improve) ≥ 0.90 (reference decision: `multi`, tuned on calib). E5 additionally rejects any change where a stratum with ≥25 entities drops > 0.02 F0.5. E2r additionally requires no-match FP ≤ binary.
- Decision oracle: the default `cal:multi` is replaced only by a non-collapsing mechanism that beats it with P(improve) ≥ 0.90.
