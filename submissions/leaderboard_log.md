# Leaderboard submission log

**No leaderboard submission has been made.** The official dataset was unavailable as of 2026-09-25.
Fill in one row per upload (cap: 5 per day, about 15 in total). Never upload a file that has not passed
`utils/validate_submission.py`. `src/pipeline/predict.py` enforces that as a hard gate in REAL mode.

Budget plan: keep at least 1 upload per day in reserve for a fix. Decide what to upload from the local
train-holdout macro F0.5, not from leaderboard probing.

| # | date/time (IST) | day | code state (hash/tag) | model cfg | decision (mechanism, params, calibrated) | matching_results.tsv sha256 (first 12) | official validator | local holdout macro F0.5 (train split) | public LB F0.5 | notes / why this upload |
|---|---|---|---|---|---|---|---|---|---|---|
| – | – | – | – | – | – | – | – | – | – | – |

Helper to fill the hash column:
```bash
python -c "import hashlib;print(hashlib.sha256(open('output/matching_results.tsv','rb').read()).hexdigest()[:12])"
```
