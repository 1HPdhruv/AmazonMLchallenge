# Phase 9 gate: dense retrieval and cross-encoder

**SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE**

## Dense retrieval (Section 12)
- Criterion 1: sparse union recall < 0.95? VAL union recall = 0.9809 -> **NOT met**
- Criterion 2 (inspection of retrieval misses in reports/error_analysis.md): {'acronym_or_nonlexical': 4, 'missing_name': 6, 'lexically_similar_but_outranked': 3}. Acronym misses are non-lexical by construction of the synthetic generator (3% of names). Missing-name misses cannot be recovered by any name-based retriever, dense or sparse. 'Lexically similar but outranked' = chain saturation: identically named branches fill the top-k (debugged example: cosine 0.914, rank 27 of k=20), usually for records with no address, which cannot be assigned to one branch anyway. A dense retriever would saturate the same way.
- **Decision: NO-GO**: criterion 1 fails, so criteria 3-4 are not evaluated and dense retrieval was not built.

## Cross-encoder (Section 13)
- Criterion 1: sparse recall target met: True
- Criterion 2: share of misclassified TEST entities whose best score lies in the ambiguous band [0.2, 0.8]: 33/137 = 24.1% (threshold 15-20%)
- Error-class mix of misclassified entities: {'multi_under_selection': 66, 'decision_miss': 33, 'retrieval_miss': 19, 'over_selection': 9, 'no_match_false_positive': 7, 'model_miss': 3}
- Criterion 3 (model size/license/reproducibility): not assessed, because no official constraints were available (dataset unavailable).
- **Decision: DEFERRED, NOT BUILT**: criteria 1-2 are met on SYNTHETIC data, but criterion 3 cannot be verified without the official rules, and a pretrained reranker would need a model download that was not approved. Re-evaluate on real data; the ambiguous-band errors are mostly `decision_miss` (single-match entities scored just under the threshold).

LLM-based matching: excluded entirely (Section 13), not evaluated.
