"""Streaming audit of the official TSV files (no pandas; constant memory).

    python tools/raw_file_audit.py Dataset

Reports per file: data rows, bad field counts, id-prefix violations, duplicate ids, empty fields,
quote characters, CR line endings, country distribution; for the ground truth: match cardinality and
referential integrity against the train source files. Writes reports/raw_file_audit.json.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

PREFIX = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}


def audit_source(path, prefix, keep_ids: set | None):
    st = collections.Counter()
    countries = collections.Counter()
    ids = set() if keep_ids is not None else None
    dup = 0
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline()
        st["crlf_header"] = int(header.endswith("\r\n"))
        for line in f:
            if line.endswith("\r\n"):
                st["crlf"] += 1
            line = line.rstrip("\r\n")
            if not line:
                st["blank_lines"] += 1
                continue
            st["rows"] += 1
            parts = line.split("\t")
            if len(parts) != 4:
                st[f"fields_{len(parts)}"] += 1
                parts = (parts + ["", "", "", ""])[:4]
            eid, name, addr, cc = parts
            if not eid.startswith(prefix):
                st["bad_prefix"] += 1
            if not name.strip():
                st["empty_name"] += 1
            if not addr.strip():
                st["empty_address"] += 1
            if not cc.strip():
                st["empty_country"] += 1
            if '"' in line:
                st["has_quote_char"] += 1
            countries[cc] += 1
            if ids is not None:
                if eid in ids:
                    dup += 1
                ids.add(eid)
    st["duplicate_ids"] = dup
    return dict(st), dict(countries.most_common(10)), ids


def audit_gt(path, s1_ids, rec_ids):
    st = collections.Counter()
    card = collections.Counter()
    linked = collections.Counter()
    seen = set()
    with open(path, encoding="utf-8", newline="") as f:
        f.readline()
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            st["rows"] += 1
            parts = line.split("\t")
            e = parts[0]
            ms = [m.strip() for m in parts[1].split(",")] if len(parts) > 1 and parts[1].strip() else []
            if e in seen:
                st["duplicate_s1_rows"] += 1
            seen.add(e)
            if e not in s1_ids:
                st["unknown_s1"] += 1
            card[min(len(ms), 10)] += 1
            if len(ms) != len(set(ms)):
                st["dup_within_list"] += 1
            for m in ms:
                st["pairs"] += 1
                linked[m] += 1
                if not m.startswith(("S2-", "S3-")):
                    st["bad_match_prefix"] += 1
                if m not in rec_ids:
                    st["unknown_record"] += 1
    st["s1_missing_from_gt"] = len(s1_ids - seen)
    st["records_linked_to_multiple_s1"] = sum(1 for v in linked.values() if v > 1)
    return dict(st), {str(k if k < 10 else "10+"): v for k, v in sorted(card.items())}


def main(root):
    t0 = time.time()
    out = {}
    for split in ("train", "test"):
        d = os.path.join(root, split.capitalize() if os.path.isdir(os.path.join(root, split.capitalize())) else split)
        ids = {}
        for src, pre in PREFIX.items():
            p = os.path.join(d, f"{split}_{src}.tsv")
            st, cc, id_set = audit_source(p, pre, set() if split == "train" else set())
            ids[src] = id_set
            out[f"{split}_{src}"] = {"stats": st, "countries": cc}
            print(f"{split}_{src}: {st} countries={cc}", flush=True)
        if split == "train":
            gst, card = audit_gt(os.path.join(d, "train_ground_truth.tsv"), ids["source1"], ids["source2"] | ids["source3"])
            out["train_ground_truth"] = {"stats": gst, "match_cardinality": card}
            print(f"train_ground_truth: {gst} cardinality={card}", flush=True)
        del ids
    out["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs("reports", exist_ok=True)
    with open("reports/raw_file_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("runtime_s", out["runtime_s"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Dataset")
