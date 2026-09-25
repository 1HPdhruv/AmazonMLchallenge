# Confirmed competition facts (Amazon ML Challenge 2026)

Recorded exactly as given in the verified operator instruction. The official PDFs were
cross-checked on 2026-09-25 and found consistent:
`D:\Users\Admin\Downloads\6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` and
`...\6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`.
The official `student_resource` dataset bundle was NOT present in the workspace when this was recorded.

- Source files (train and test): entity_id (prefixed S1-/S2-/S3-), business_name,
  business_address, country. Source identity comes from the entity_id prefix and
  which file the record is in — there is no separate source column.
- All files are TAB-SEPARATED (.tsv), including our own output files.
- train_ground_truth.tsv: source1_entity_id, matched_entity_ids (comma-separated
  S2-/S3- ids, empty = no match). A Source-1 entity may have zero, one, or many
  matches.
- country is an OPEN SET: train = {US, India}; test adds a THIRD country, France,
  absent from training. Every test Source-1 entity, France included, must appear
  in the submission. Never hardcode/one-hot to {US, India}.
- Evaluation formula (exact): F_0.5 = (1.25 x Precision x Recall) /
  (0.25 x Precision + Recall), computed per Source-1 entity, then macro-averaged
  across all Source-1 entities. A no-match entity scores 1.0 for a correctly-empty
  prediction, 0.0 for any predicted match (no partial credit).
- Required outputs: output/matching_results.tsv (source1_entity_id,
  matched_entity_ids — the ONLY file scored on the leaderboard) AND
  output/candidate_pairs.tsv (source1_entity_id, candidate_entity_ids — the exact
  candidate set fed to the final scoring stage). predicted matches must be a
  SUBSET of candidates. Every test Source-1 entity needs exactly one row in both
  files; no duplicate ids, no self-matches to S1, no ids outside the test set.
- Official validator: utils/validate_submission.py (stdlib only) — run before
  every leaderboard upload; must be wired in as a hard gate once present.
- Challenge window: 25 Sept 2026 00:00 IST -> 27 Sept 2026 23:59 IST (today is
  day 1 of 3).
- Submission cap: 5 leaderboard submissions/day; submit button disables after
  day 3 (~15 total, ever).
- Model constraint (exact): final model must be MIT or Apache-2.0 licensed,
  <=8 billion parameters. Our GBDT backbone trivially satisfies this.
- No external entity/business lookups, APIs, registries, or geocoding, ever.
- Final submission zip structure (exact):
    <team>_submission.zip
      output/matching_results.tsv
      output/candidate_pairs.tsv
      code/business_entity_resolution/src/
      code/business_entity_resolution/README.md
      code/business_entity_resolution/requirements.txt
      Documentation_template.md   (filled in: methodology, blocking/candidate-
        generation strategy, model architecture, feature engineering; no page
        limit for this full package version)

## Additional details from the official problem statement (cross-check)
- Expected layout: `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`,
  `dataset/test/test_source{1,2,3}.tsv`; read with `sep="\t"`.
- Output example: columns separated by a single tab; ID lists comma-separated with **no quoting**;
  an empty second column for no-match / no-candidate entities.
- `candidate_pairs.tsv` is the *last* blocking/filtering stage, i.e. whatever the model actually runs inference over.
- Validator invocation (from `student_resource/`):
  `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
  → prints PASS and exits 0, or lists the issues and exits 1.
- The guidelines PDF also asks for a 1–2 page approach document for the best solution, separate from the full
  package's `Documentation_template.md`, which has no page limit.
