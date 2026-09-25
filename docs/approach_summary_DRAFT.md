# Approach summary (1-2 pages): Business Entity Resolution

> **DRAFT.** Every result is a `TODO(real-data)` placeholder. Nothing here reports a competition number.

**Problem.** Link each Source-1 business (the deduplicated reference) to every Source-2/Source-3 record that
describes the same business. The link can be to none, one or many records, using only name, free-text address
and country. Scoring is per-entity F0.5, macro-averaged: precision counts double, and one false match on a
no-match entity costs the whole entity. The test set adds France, which does not appear in training.

**Approach: retrieve, score, decide, each stage measured separately.**
1. **Normalisation.** Conservative and additive: the raw text is kept, and alongside it we build casefolded,
   accent-folded, abbreviation-expanded and legal-form-aware versions, collapse dotted initials, and parse
   address components by content. The parser recognises landmark phrases, units, and postal codes anywhere.
2. **Candidate generation.** A union of four sparse channels: an exact name key, char 3-gram TF-IDF, word TF-IDF,
   and rarity-weighted address keys. IDF is fit on training entities only. Each channel's recall and candidate
   volume are measured, and a channel stays only if its marginal recall justifies its volume. The candidate set
   written to `candidate_pairs.tsv` is exactly what the model scores. Recall ceiling: `TODO(real-data)`.
3. **Pair model.** Gradient-boosted trees (XGBoost) over lexical, information-content (IDF-weighted),
   component-level address, and context features. Class imbalance is handled with `scale_pos_weight`.
4. **Entity-level decision.** Scores are isotonically calibrated on a held-out calibration fold. The top-1 is
   accepted only above a threshold, and extra matches only if they are near the top score. Because Source 1 is
   deduplicated, each record goes to at most one entity: its strict best. Thresholds are tuned directly for
   macro F0.5.

**Experiments and evidence discipline.**
- Entity-level splits: train-fit, calibration, validation, holdout.
- Every change was tested alone against the current best, and kept only with a paired-bootstrap
  P(improve) ≥ 0.9.
- A leave-one-country-out drill (US↔India) probes zero-shot behaviour before France is seen.
- Tested and not kept, with reasons in the experiment log: ranking objective, entity weighting, hard negatives,
  Fellegi-Sunter weight, candidate-relative features, phonetic channel, parse-free address channel, legal-form
  dictionary, unseen-country threshold margin. Real-data outcomes: `TODO(real-data)`.

**Results.** `TODO(real-data)`: validation macro F0.5, precision, recall, no-match false-positive rate,
per-country breakdown, and public leaderboard scores for each submission.

**Conclusions.** `TODO(real-data)`.

**Why no LLMs.** Excluded for reproducibility, the MIT/Apache ≤ 8B final-model rule, cost at candidate-set
scale, and the risk that LLM world knowledge about businesses counts as external lookup. The measured error
sources are structural (recall ceiling, singleton false positives, multi-match cardinality), not semantic.
