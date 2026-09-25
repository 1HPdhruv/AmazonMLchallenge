"""Official output files, local rule checks, and the official-validator HARD GATE.

Contract (configs/competition_facts.md):
  output/matching_results.tsv : source1_entity_id <TAB> matched_entity_ids    (the only scored file)
  output/candidate_pairs.tsv  : source1_entity_id <TAB> candidate_entity_ids  (exact set the model scored)
  - one row per test Source-1 entity, in both files; ID lists comma-separated, no quoting;
    empty list for no match / no candidates
  - only S2-/S3- ids that exist in the test files; no duplicates within a list; no duplicate rows
  - matches must be a SUBSET of candidates (enforced by assertion before anything is written)
"""
from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd

from src.evaluation.entity_metrics import fbeta

MATCH_FILE, CAND_FILE = "matching_results.tsv", "candidate_pairs.tsv"
MATCH_COLS = ["source1_entity_id", "matched_entity_ids"]
CAND_COLS = ["source1_entity_id", "candidate_entity_ids"]


class OutputContractError(Exception):
    pass


def build_lists(s1_ids, cand_pairs: pd.DataFrame, match_pairs: pd.DataFrame):
    """cand_pairs/match_pairs: columns entity_id, record_id (+ score for ordering matches)."""
    cands = cand_pairs.groupby("entity_id")["record_id"].apply(lambda x: sorted(set(x))).to_dict()
    if "score" in match_pairs:
        match_pairs = match_pairs.sort_values(["entity_id", "score"], ascending=[True, False])
    matches = match_pairs.groupby("entity_id", sort=False)["record_id"].apply(lambda x: list(dict.fromkeys(x))).to_dict()
    for e, m in matches.items():  # subset check BEFORE writing
        extra = set(m) - set(cands.get(e, []))
        if extra:
            raise OutputContractError(f"{e}: matched ids not in candidate set: {sorted(extra)[:5]}")
    return {e: cands.get(e, []) for e in s1_ids}, {e: matches.get(e, []) for e in s1_ids}


def write_outputs(out_dir, s1_ids, cand_lists: dict, match_lists: dict):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for fname, cols, lists in ((CAND_FILE, CAND_COLS, cand_lists), (MATCH_FILE, MATCH_COLS, match_lists)):
        p = os.path.join(out_dir, fname)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write("\t".join(cols) + "\n")
            for e in s1_ids:
                f.write(f"{e}\t{','.join(lists[e])}\n")
        paths[fname] = p
    return paths


def read_output(path, cols):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            line = line.rstrip("\r\n")
            if line == "":
                continue
            parts = line.split("\t")
            rows.append((parts[0], parts[1] if len(parts) > 1 else "", len(parts)))
    return header, rows


def local_checks(out_dir, test_s1_ids, test_rec_ids: set) -> list[str]:
    """Mirror of every documented rule (the official validator stays authoritative)."""
    issues = []
    s1 = list(test_s1_ids)
    lists = {}
    for fname, cols in ((MATCH_FILE, MATCH_COLS), (CAND_FILE, CAND_COLS)):
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            issues.append(f"{fname}: missing")
            continue
        header, rows = read_output(p, cols)
        if header != cols:
            issues.append(f"{fname}: header {header} != {cols}")
        ids = [r[0] for r in rows]
        if any(n > 2 for *_, n in rows):
            issues.append(f"{fname}: rows with more than 2 tab-separated fields")
        if len(ids) != len(set(ids)):
            issues.append(f"{fname}: duplicate source1_entity_id rows")
        if set(ids) != set(s1):
            issues.append(f"{fname}: {len(set(s1) - set(ids))} test S1 entities missing, "
                          f"{len(set(ids) - set(s1))} unknown S1 ids")
        d = {}
        for e, v, _ in rows:
            lst = [x for x in v.split(",")] if v else []
            if any(x == "" or x != x.strip() for x in lst):
                issues.append(f"{fname}: {e}: empty/whitespace id in list")
            if len(lst) != len(set(lst)):
                issues.append(f"{fname}: {e}: duplicate ids in list")
            bad = [x for x in lst if not x.startswith(("S2-", "S3-"))]
            if bad:
                issues.append(f"{fname}: {e}: non S2-/S3- ids (self-match?) {bad[:3]}")
            unknown = [x for x in lst if x not in test_rec_ids]
            if unknown:
                issues.append(f"{fname}: {e}: ids not in test set {unknown[:3]}")
            d[e] = set(lst)
        lists[fname] = d
    if MATCH_FILE in lists and CAND_FILE in lists:
        viol = [e for e, m in lists[MATCH_FILE].items() if not m <= lists[CAND_FILE].get(e, set())]
        if viol:
            issues.append(f"matches not a subset of candidates for {len(viol)} entities, e.g. {viol[:3]}")
    return issues[:50]


def official_gate(base: dict, out_dir: str, allow_missing: bool = False) -> dict:
    """Runs utils/validate_submission.py (stdlib only) if it exists. HARD GATE: raises on failure.
    In REAL mode a missing validator is also an error unless allow_missing=True."""
    root = base["data"]["root"]
    v = os.path.join(root, base["data"].get("official_validator", "utils/validate_submission.py"))
    if not os.path.exists(v):
        if base["mode"] == "real" and not allow_missing:
            raise OutputContractError(f"official validator not found at {v}; refusing to mark output as submittable")
        return {"ran": False, "passed": None, "note": f"official validator not present ({v})"}
    cmd = [sys.executable, os.path.abspath(v), "--matching", os.path.abspath(os.path.join(out_dir, MATCH_FILE)),
           "--candidate", os.path.abspath(os.path.join(out_dir, CAND_FILE)), "--test-dir", os.path.join("dataset", "test")]
    res = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    out = {"ran": True, "passed": res.returncode == 0, "cmd": " ".join(cmd), "stdout": res.stdout[-4000:],
           "stderr": res.stderr[-2000:]}
    if res.returncode != 0:
        raise OutputContractError(f"OFFICIAL VALIDATOR FAILED (exit {res.returncode}):\n{res.stdout}\n{res.stderr}")
    return out


def score_matching(path, labels: pd.DataFrame, s1_ids) -> float:
    """Macro F0.5 of a matching_results.tsv against labels (synthetic hidden labels / train holdout only)."""
    _, rows = read_output(path, MATCH_COLS)
    pred = {e: frozenset(v.split(",")) if v else frozenset() for e, v, _ in rows}
    truth = labels.groupby("entity_id")["record_id"].apply(frozenset).to_dict()
    return sum(fbeta(pred.get(e, frozenset()), truth.get(e, frozenset())) for e in s1_ids) / len(s1_ids)


def write_report(path, label, stats: dict, issues: list, gate: dict, synthetic_score=None):
    lines = ["# Output validation report\n"]
    if label:
        lines.append(f"**{label}**\n")
        lines.append("**No official competition submission was generated because the actual competition dataset "
                     "is unavailable.** These files were produced from the SYNTHETIC test tree to verify output "
                     "mechanics only.\n")
    lines += [f"- {k}: {v}" for k, v in stats.items()]
    lines += ["", "## Local rule checks (mirror of the documented rules)",
              "- PASS" if not issues else "\n".join(f"- FAIL: {i}" for i in issues),
              "", "## Official validator (utils/validate_submission.py): HARD GATE",
              f"- ran: {gate.get('ran')}; passed: {gate.get('passed')}; {gate.get('note', '')}"]
    if gate.get("stdout"):
        lines += ["```", gate["stdout"], "```"]
    if synthetic_score is not None:
        lines += ["", f"## {label}", f"- macro F0.5 of output/matching_results.tsv vs synthetic hidden labels: "
                                     f"{synthetic_score:.4f}"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
