# Deep-improvement pass: audit, ablations, checkpoint

**SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE.** The real dataset is still missing. Every number below
comes from the synthetic benchmark (official layout; train = US/India, test adds France) and none is a
leaderboard estimate. The full per-variant table is in `reports/ablation_log.md`.

## 1. Adversarial audit: ranked, verified findings
Method:
- Six code-area auditors plus a literature/feasibility researcher ran in a multi-agent workflow.
- Every finding was sent to two skeptics, one checking the mechanism and one checking the impact.
- The workflow hit a usage limit partway through. Four auditors completed (24 findings); the features/model
  auditor, the evaluation auditor, the research agent and some skeptics did not.
- Every finding listed here was then re-verified by hand against the current code with snippets (evidence
  column).

| # | finding (mechanism) | harm | evidence | synthetic-measurable |
|---|---|---|---|---|
| 1 | The global LEGAL/STOP sets delete real name tokens anywhere in a name: `AB Enterprises`→key `enterprises`, `Das Brothers & Co`→`brothers`. The exact channel then pairs distinct firms, with name_key_eq=1 and word_cos=1. | false merges | snippet | no (the generator never emits such names) |
| 2 | Tokens unseen in US/India train get the **maximum** IDF (`tfidf.py`), so French generic words (`societe`) and legal-form trigrams (SARL/SASU) look like brand-strength evidence. | France / false merges | code + snippet | partly (France slice) |
| 3 | Record exclusivity was shipped OFF, although Source 1 is officially deduplicated and train GT has 0 records linked to >1 entity. When on, it only arbitrated among already-accepted pairs and kept ties. | false merges | code; GT statistic | yes |
| 4 | `tune()` keeps the first (most permissive) threshold on a calib-F0.5 plateau. | no-match FP | code | yes |
| 5 | The positional address parser: a landmark-first segment becomes the street (house number lost, real street stored as city); a `, Suite 120` segment becomes the city; there is no order-free address evidence. | recall ceiling / precision | snippet (`Near SBI ATM, 12 MG Road...` → house None) | yes (the generator emits landmarks) |
| 6 | Top-k ties are cut by row order, so Source 2 records win ties. | recall ceiling | Source-3 retrieval-miss rate 8.2% vs 1.9% for Source 2 | yes, **causal claim refuted** (R1) |
| 7 | Chain saturation: name-only top-k has no location term. | recall ceiling | earlier debug (cos 0.914 ranked 27th of k=20) | yes |
| 8 | Secondary matches are gated relative to the top score; the absolute rule `multi_t2` only allowed t2 ≤ t. | multi-match recall | code | yes |
| 9 | French street core keeps its article: keys degrade to `hs|12|de`, and Jaro-Winkler between different `de ...` streets is inflated. | France | snippet | yes |
| 10 | The postal regex misses ZIP+4, spaced or glued Indian PINs and `F-75008`; `CA`/`IN`/`DE` state codes are stripped as countries; dotted initials split into single letters. | real formats | snippets | no (unit tests only) |
| 11 | No country conditioning in blocking (French postal codes collide with US ZIPs; cross-country top-k competition). | recall / volume | code | no, deferred to real data |
| 12 | Calibration and threshold are fit on US/India only, with no unseen-country guard. | France | LOCO drill: unseen country needs about +0.21 t | yes |

## 2. What was implemented, kept and rejected
Keep rules were fixed before any run:
- **Measurable change:** VAL P(improve) ≥ 0.9.
- **France-targeted change:** DEV-France P(improve) ≥ 0.9 and VAL P(improve) ≥ 0.2.
- **Real-format bug fix:** unit tests pass and no regression.

| id | change | result (Δ macro F0.5, paired bootstrap) | decision |
|---|---|---|---|
| D1 | precision-leaning tie-break in `tune()` | VAL +0.000 (plateau genuinely flat) | rejected |
| D2a | exclusivity among accepted pairs | VAL +0.0045 (P 0.999) | superseded by D2b |
| **D2b** | **exclusivity over all candidates, ties dropped** | **VAL +0.0119 [0.0069, 0.0177]; France +0.0109 (P 0.986); VAL no-match FP 0.093→0.059** | **kept** (ships as `exclusive_mode: auto`) |
| D3 | absolute secondary bar (`abs2`, t2 may exceed t1) | VAL +0.0001 (P 0.53) | rejected |
| N1/N1a/N1c | per-country legal-form dictionary (user item a) | VAL −0.002 / −0.006 / −0.0045; France +0.004 / −0.002 / −0.006 | rejected on synthetic (see note) |
| N2 | unseen-token IDF → train median + unseen-count features | VAL +0.0007; France +0.0021 (P 0.74) | rejected |
| **A1** | **address parser v2** (landmarks, unit segment, content-chosen street, postal anywhere, FR articles/types, state guard) | **VAL recall 0.962→0.974; VAL +0.0036 (P 0.84); France +0.0095 (P 0.958)** | **kept** (France rule and bug-fix rule) |
| R1 | keep tied block at the top-k cutoff | +33% pairs, no recall gain, VAL +0.0025 (P 0.80) | rejected; refutes finding #6's causal claim |
| R2f / R2 | parse-free address TF-IDF feature / channel | VAL −0.0033 / −0.0012 (+40% pairs) | rejected |
| R3 | `name_geo`: name top-100 re-scored by address (chain saturation) | VAL +0.0006, +24% pairs | rejected |
| P1f / P1c | phonetic (Metaphone + NYSIIS) features / exact channel (user item b) | VAL −0.0032 / −0.0011; P1c recall +0.004 at +0.7% pairs | rejected on F0.5; P1c flagged for real data |
| **B1** | **dotted initials / `M/s` collapse (bug fix)** | **byte-identical synthetic output (no such inputs); unit tests pass** | **kept** |
| F1 | unseen-country threshold margin from the LOCO drill | France no-match FP 0.120→0.083, precision +0.009, recall −; France F0.5 ±0.000 | rejected (net zero) |

**libpostal (user item c): not vendored.**
- The model data is about 1.8 GB (2.2 GB for the alternative model), against about 1–2 GB of free RAM here.
- The Windows build needs MSYS2/MinGW and has no Python 3.13 wheels.
- The parser is trained on OpenStreetMap/OpenAddresses under ODbL, which risks both the MIT/Apache final-model
  requirement and the "no external data augmentation" rule.
- Instead, the targeted parser fixes of A1 addressed the specific failures the audit found.

**Blocking-literature refinements (user item d).** Evaluated through the retrieval gate:
- Splink-style conjunctive name AND place rule (R3).
- Order-free address predicates in the style of Dedupe (R2).
- Tie-block retrieval (R1).
- A cheap phonetic key channel (P1c).

None improved F0.5. Dense retrieval (SC-Block/UniBlocker) stays gated: union recall is 0.974 ≥ 0.95.

**Note on the legal dictionary.** The synthetic generator builds names from the same global suffix list the
legacy normaliser strips, so the benchmark structurally favours the legacy behaviour. It also cannot express the
precision hazard in finding #1. The dictionary (`configs/legal_forms.yaml` + `src/preprocessing/legal.py`, with
unit tests) ships **off** and is the first real-data A/B to run.

## 3. Checkpoint status
- **Best synthetic checkpoint (v2):** B0 → D2b → A1 → B1, preserved in `artifacts/synthetic_best_v2/`. The
  shipped `configs/` reproduce it exactly: the production `train` + `predict` path gives DEV macro F0.5
  0.9348, identical to ablation B1.
- **Untouched holdout2** (fresh synthetic test tree; no selection used it):
  - Overall: 0.9109 (B0) → **0.9361** (B1), +0.0252, 90% CI [0.0188, 0.0318].
  - France: 0.911 → 0.926 (+0.0149, P 0.991). India: 0.910 → 0.943. US: 0.911 → 0.936.
  - No-match FP: 0.118 → 0.069. Pair precision: 0.914 → 0.957. Candidate recall: 0.967 → 0.976.
- **v1** (pre-compliance schema) remains in `artifacts/synthetic_best_v1/`. It isn't comparable, because its
  data had structured addresses and phone numbers.
- **Regression:** 46/46 tests pass in the clean `.venv` (new: `tests/unit/test_deep_pass.py`). The packager
  builds the exact zip structure and refuses synthetic output. The run is deterministic across processes.
