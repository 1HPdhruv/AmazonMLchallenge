# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

> **DRAFT.** The official `Documentation_template.md` ships inside the `student_resource` bundle, which was
> not available when this was written. Move these sections into the official template's headings once it
> arrives. Every result is a `TODO(real-data)` placeholder. No number here comes from the competition data,
> and synthetic-benchmark numbers are deliberately left out.

## 1. Problem framing
- **Task.** For every Source-1 entity (the deduplicated reference source), find the set of Source-2/Source-3
  records that describe the same real-world business. The set can be empty, one record, or many records.
- **Inputs.** Only the four supplied fields: `entity_id` (the S1-/S2-/S3- prefix gives the source), `business_name`,
  `business_address` (one free-text field) and `country`. No external data, lookup, registry or geocoding is used.
- **Metric and what it implies.** The metric is F0.5 per Source-1 entity, macro-averaged over entities.
  - Precision is weighted more heavily than recall.
  - A no-match entity scores 0 for *any* predicted match.
  - A false merge therefore costs about twice a miss, and much more on singletons. Every stage is designed with
    that asymmetry in mind.
- **Formulation.** The system has three stages with separate measurements:
  1. Candidate generation, which sets the recall ceiling.
  2. Pairwise scoring.
  3. An entity-level decision, which chooses zero, one or many matches per entity.
- **Generalisation requirement.** The test set contains France, which is absent from training. Country is
  treated as an open-set string: nothing is one-hot encoded or filtered to {US, India}. Every test entity,
  France included, gets a row in both output files.

## 2. Validation methodology
- Splits are made at the entity level on the labelled train files: `train_fit` / `calib` / `val` / internal holdout.
  All candidate pairs of a Source-1 entity share its split, so there is no pair-level leakage.
- Statistics are fit on `train_fit` only: TF-IDF IDF weights, address-key IDF, the GBDT, and hard-negative
  selection. Isotonic calibration and decision thresholds are fit on `calib`. Component choices are made on
  `val` with a pre-registered paired-bootstrap rule (keep a change only if P(improve) ≥ 0.9). The internal
  holdout is used once, for reporting.
- Zero-shot check for the unseen country: a leave-one-country-out drill inside train (train on US, evaluate on
  India, and the reverse) estimates how decisions shift for a country the model has never seen.
- Results: `TODO(real-data)`: val / holdout macro F0.5, precision, recall, no-match false-positive rate, and
  per-country results.

## 3. Candidate generation (blocking)
**Rationale.** Blocking sets the recall ceiling, and no single rule survives every noise pattern in the problem
statement (abbreviations, legal-suffix changes, typos, transliteration, reordered or missing address
components, landmark references). We therefore take the union of independent, sparse, recall-oriented
channels, measure each channel's recall and candidate volume separately, and keep a channel only if its
marginal recall is worth its volume cost. This follows the union-of-blocking-rules methodology of
Splink/Dedupe and top-k TF-IDF blocking (Sparkly).

| channel | what it retrieves | why |
|---|---|---|
| exact key | identical order-insensitive cleaned name | cheap, high-precision anchor |
| char 3-gram TF-IDF top-k | names within typo/abbreviation distance | robust to spelling noise and missing spaces |
| word TF-IDF top-k | names sharing rare tokens | IDF weighting keeps generic words ("Group", "Services") from dominating |
| address keys | records sharing rare postal / house+street / street+city keys | recovers pairs whose names differ (DBA / trade names) |

- IDF weights are fit on training entities only (hashed features, so there is no vocabulary leakage).
- Address keys come from a content-based parser that sets aside landmark phrases, detects postal codes anywhere
  in the string (ZIP+4, spaced or glued Indian PINs, French codes), and strips French articles. Keys whose
  train IDF falls below a floor, or whose posting list is too long, are dropped, so common house numbers never
  fire alone.
- `output/candidate_pairs.tsv` is exactly the union passed to the model. Predicted matches are *asserted* to be
  a subset of it before anything is written.
- Per-channel recall and volume: `TODO(real-data)`. Dense retrieval was gated on sparse union recall < 0.95:
  `TODO(real-data)` decision.

## 4. Pair model and feature engineering
- **Model.** Gradient-boosted trees (XGBoost, Apache-2.0; about 400 trees, far below 8B parameters), binary
  objective, class imbalance handled with `scale_pos_weight`. We chose binary classification over a ranking
  objective because ranking gives no learning signal for no-match entities, which are exactly where F0.5 is
  least forgiving.
- **Feature groups:**
  - *Lexical (name):* Levenshtein ratio, Jaro-Winkler, token sort/set ratios, partial ratio, Jaccard,
    containment, length ratio, first-token agreement, char and word TF-IDF cosine, and canonical legal-form
    agreement.
  - *Information content:* shared-token IDF sum and maximum, rare shared tokens, IDF-weighted Jaccard. Agreeing
    on a rare token is strong evidence; agreeing on a generic one is weak (the Fellegi-Sunter intuition).
  - *Address (component-level):* house number, unit, postal and postal-prefix agreement, street similarity,
    city and region agreement, numeric-token overlap, address-token containment and set similarity.
  - *Context:* country agreement, source of the record, missingness indicators, and per-channel retrieval
    ranks and scores.
- Hard-negative mining, entity weighting, a Fellegi-Sunter match weight, candidate-relative features, phonetic
  features and parse-free address TF-IDF were all built and ablated. Their keep or reject decisions are
  `TODO(real-data)`, re-run on the real train split.

## 5. Entity-level decision layer (precision over recall)
- Pair scores are isotonically calibrated on the `calib` fold. Then, per Source-1 entity:
  - The top candidate is accepted only if its calibrated score clears a threshold `t`.
  - Further candidates are accepted only if they also clear `t` and are within `δ` of the top score. This
    supports multi-match without letting weak siblings through.
- **Record exclusivity.** Source 1 is deduplicated, so a Source-2/3 record can belong to at most one Source-1
  entity. A record is kept only for its strict best-scoring entity over *all* candidates, and exact ties are
  dropped. This removes the classic false merge in which one record is claimed by several branches of a chain.
  It is enabled automatically only if the train ground truth confirms that no record is linked to two entities.
- `t` and `δ` are tuned on `calib` to maximise macro F0.5 directly, not pairwise accuracy. The final values
  are `TODO(real-data)`.
- **Why this shape.** Under macro F0.5 a wrong accept on a singleton costs a full point. Accepting a weak top-1
  pays off only when its probability of being correct is high. Adding a secondary match to an entity whose
  top-1 is already correct is worth it only when the secondary is very likely right (break-even ≈ 0.73 for two
  true matches).

## 6. Why LLM-based approaches were excluded
- **Reproducibility:** generative inference is non-deterministic across versions and hardware. The final output
  must be regenerable from the supplied data and code alone.
- **Licensing and size:** the final model must be MIT/Apache-2.0 and ≤ 8B parameters. The whole pipeline
  (sparse retrieval + GBDT) meets this trivially and needs no pretrained weights.
- **Scale and precision:** pairwise LLM scoring over a large candidate set is expensive and still needs the same
  calibrated, entity-level, precision-first decision layer. The measured error sources (recall ceiling,
  singleton false positives, multi-match cardinality) are structural, not semantic.
- **Rule risk:** LLM world knowledge about real businesses can amount to external entity knowledge, which the
  rules forbid.

## 7. Reproducibility
- `code/business_entity_resolution/README.md` gives the exact commands. `requirements.txt` pins every version.
  The runs are deterministic: identical outputs across processes and hash seeds, covered by a test.
- `utils/validate_submission.py` is a hard gate before every leaderboard upload.

## 8. Results
`TODO(real-data)`: leaderboard submissions table, local validation vs public leaderboard, per-country breakdown,
error analysis (retrieval miss / model miss / decision miss / no-match false positive / multi-match
under- and over-selection).
