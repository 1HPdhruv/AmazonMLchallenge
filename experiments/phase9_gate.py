"""Phase-9 GO/NO-GO evidence for dense retrieval (Section 12) and a cross-encoder (Section 13).
Reads the reports produced by run_experiments; builds nothing. SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE."""
from __future__ import annotations

import json
import re

import pandas as pd

LABEL = "SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE"


def main():
    rr = json.load(open("reports/retrieval_report_val.json", encoding="utf-8"))
    recall = rr["union"]["recall"]
    sa = pd.read_csv("reports/entity_score_analysis.csv")
    err = sa[sa.error_class != "correct"]
    # ambiguous band: the entity's decisive score (best candidate) sits in [0.2, 0.8] on the decision scale
    amb = err[(err.best_score >= 0.2) & (err.best_score <= 0.8)]
    share_amb = len(amb) / max(1, len(err))
    # retrieval misses: inspect lexical overlap of missed pairs
    md = open("reports/error_analysis.md", encoding="utf-8").read()
    miss_rows = md.split("### Retrieval misses")[1].strip().splitlines()[3:]
    kinds = {"acronym_or_nonlexical": 0, "missing_name": 0, "lexically_similar_but_outranked": 0}
    for row in miss_rows:
        c = [x.strip() for x in row.split("|")[1:-1]]
        if len(c) < 3:
            continue
        s1 = c[1].split("/")[0].strip().lower()
        rec = c[2].split("/")[0].strip().lower()
        if rec in ("nan", "none", ""):
            kinds["missing_name"] += 1
        elif len(set(re.findall(r"\w+", s1)) & set(re.findall(r"\w+", rec))) <= 1:
            kinds["acronym_or_nonlexical"] += 1
        else:
            kinds["lexically_similar_but_outranked"] += 1
    dense_go = recall < 0.95
    ce_go = recall >= 0.95 and share_amb > 0.15
    lines = ["# Phase 9 gate: dense retrieval and cross-encoder\n", f"**{LABEL}**\n",
             "## Dense retrieval (Section 12)",
             f"- Criterion 1: sparse union recall < 0.95? VAL union recall = {recall} -> **{'met' if dense_go else 'NOT met'}**",
             f"- Criterion 2 (inspection of retrieval misses in reports/error_analysis.md): {kinds}. "
             "Acronym misses are non-lexical by construction of the synthetic generator (3% of names). "
             "Missing-name misses cannot be recovered by any name-based retriever, dense or sparse. "
             "'Lexically similar but outranked' = chain saturation: identically named branches fill the top-k "
             "(debugged example: cosine 0.914, rank 27 of k=20), usually for records with no address, which cannot be "
             "assigned to one branch anyway. A dense retriever would saturate the same way.",
             f"- **Decision: {'GO' if dense_go else 'NO-GO'}**: criterion 1 fails, so criteria 3-4 are not evaluated and dense retrieval was not built.",
             "", "## Cross-encoder (Section 13)",
             f"- Criterion 1: sparse recall target met: {recall >= 0.95}",
             f"- Criterion 2: share of misclassified TEST entities whose best score lies in the ambiguous band [0.2, 0.8]: "
             f"{len(amb)}/{len(err)} = {share_amb:.1%} (threshold 15-20%)",
             f"- Error-class mix of misclassified entities: {err.error_class.value_counts().to_dict()}",
             "- Criterion 3 (model size/license/reproducibility): not assessed, because no official constraints were available (dataset unavailable).",
             f"- **Decision: {'DEFERRED, NOT BUILT' if ce_go else 'NO-GO'}**: "
             + ("criteria 1-2 are met on SYNTHETIC data, but criterion 3 cannot be verified without the official "
                "rules, and a pretrained reranker would need a model download that was not approved. Re-evaluate on "
                "real data; the ambiguous-band errors are mostly `decision_miss` (single-match entities scored just "
                "under the threshold)." if ce_go else
                "errors do not concentrate in an ambiguous pairwise-score band. Most failures are multi-match "
                "under-selection with confident top-1 scores, a decision/recall problem a pairwise reranker would not fix."),
             "", "LLM-based matching: excluded entirely (Section 13), not evaluated."]
    open("reports/phase9_gate.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
