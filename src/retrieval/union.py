"""Candidate union + the ONE candidate-distribution / retrieval report generator."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from src.retrieval.channels import AddressChannel, address_keys, exact_channel
from src.retrieval.tfidf import HashedTfidf, topk_cosine

CHANNELS = ["exact", "char", "word", "addr"]
EXTRA_CHANNELS = ["addr_text", "name_geo", "phon"]   # optional; their columns exist only when enabled


def _rowdot(A, B, ei, rj, chunk=50000):
    out = np.empty(len(ei), np.float32)
    for s in range(0, len(ei), chunk):
        out[s:s + chunk] = np.asarray(A[ei[s:s + chunk]].multiply(B[rj[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def active_channels(cand) -> list[str]:
    return [c for c in CHANNELS + EXTRA_CHANNELS if f"hit_{c}" in cand]


class CandidateGenerator:
    """Fits channel statistics on TRAIN entities, then retrieves for any entity set."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        uns = cfg.get("unseen_idf", {})
        self.char = HashedTfidf("char", uns.get("char", "max"))
        self.word = HashedTfidf("word", uns.get("word", "max"))
        a = cfg["address"]
        self.addr = AddressChannel(a["idf_floor"], a["max_postings"], a["k"], cfg.get("tie_cap", 0))
        self.enabled = cfg["channels"]
        self.extra = [ch for ch in EXTRA_CHANNELS if (cfg.get(ch) or {}).get("enabled")]
        # parse-free address representation: char TF-IDF over the whole folded address (IDF fit on train)
        need_addr = any(ch in ("addr_text", "name_geo") for ch in self.extra) or cfg.get("addr_text_feature")
        self.addrtxt = HashedTfidf("char") if need_addr else None
        self.runtime = {}

    def fit(self, train_ent_rep: pd.DataFrame):
        self.char_field = self.cfg.get("char_field", "name_fold")
        self.word_field = self.cfg.get("word_field", "name_clean")
        self.key_field = self.cfg.get("key_field", "name_key")
        self.char.fit(train_ent_rep[self.char_field])
        self.word.fit(train_ent_rep[self.word_field])
        self.addr.fit([address_keys(r) for _, r in train_ent_rep.iterrows()])
        if self.addrtxt is not None:
            self.addrtxt.fit(train_ent_rep["addr_fold"])
        return self

    def encode(self, rep: pd.DataFrame):
        return {"char": self.char.transform(rep[getattr(self, "char_field", "name_fold")]),
                "word": self.word.transform(rep[getattr(self, "word_field", "name_clean")]),
                "country": rep["country_n"].map(lambda v: v if isinstance(v, str) and v else "").values,
                "addr_keys": [address_keys(r) for _, r in rep.iterrows()],
                "key": rep[getattr(self, "key_field", "name_key")].tolist(),
                "phon_key": rep["name_mp_key"].tolist() if "name_mp_key" in rep else None,
                **({"addrtxt": self.addrtxt.transform(rep["addr_fold"])} if getattr(self, "addrtxt", None) is not None else {})}

    def _name_topk(self, ent_enc, rec_enc, ch):
        """Name top-k. engine 'sparse' = sparse_dot_topn (no dense blocks). partition.enabled: top-k is taken
        within the entity country (records with an empty country join every partition) plus a global
        fallback top-`fallback_k`, so a mislabelled or cross-country record stays reachable."""
        c = self.cfg
        x = c[ch]
        eng = c.get("engine", "dense")
        nt = c.get("n_threads", 0)
        part = c.get("partition") or {}
        Q, D = ent_enc[ch], rec_enc[ch]
        if not part.get("enabled"):
            return topk_cosine(Q, D, x["k"], x["min_score"], tie_cap=c.get("tie_cap", 0), engine=eng, n_threads=nt)
        qc, dc = ent_enc["country"], rec_enc["country"]
        parts = []
        for cc in sorted(set(qc)):
            qi_c = np.where(qc == cc)[0]
            di_c = np.where((dc == cc) | (dc == ""))[0] if cc else np.arange(D.shape[0])
            if len(qi_c) == 0 or len(di_c) == 0:
                continue
            a, b, s_, _ = topk_cosine(Q[qi_c], D[di_c], x["k"], x["min_score"], engine=eng, n_threads=nt)
            parts.append((qi_c[a], di_c[b], s_))
        fb = int(part.get("fallback_k", 0))
        if fb:
            a, b, s_, _ = topk_cosine(Q, D, fb, x["min_score"], engine=eng, n_threads=nt)
            parts.append((a, b, s_))
        if not parts:
            return (np.array([], int),) * 2 + (np.array([]), np.array([], int))
        d = pd.DataFrame({"e": np.concatenate([p[0] for p in parts]), "r": np.concatenate([p[1] for p in parts]),
                          "s": np.concatenate([p[2] for p in parts])})
        d = d.sort_values(["e", "s", "r"], ascending=[True, False, True]).drop_duplicates(["e", "r"])
        d["rk"] = d.groupby("e").cumcount() + 1
        return d.e.values, d.r.values, d.s.values, d.rk.values

    def retrieve(self, ent_enc, rec_enc) -> pd.DataFrame:
        c = self.cfg
        outs = []
        for ch in CHANNELS:
            if not self.enabled.get(ch, False):
                continue
            t0 = time.time()
            if ch == "exact":
                r = exact_channel(ent_enc["key"], rec_enc["key"], c["exact_max_postings"])
            elif ch == "char":
                r = self._name_topk(ent_enc, rec_enc, "char")
            elif ch == "word":
                r = self._name_topk(ent_enc, rec_enc, "word")
            else:
                r = self.addr.retrieve(ent_enc["addr_keys"], rec_enc["addr_keys"])
            self.runtime[ch] = round(time.time() - t0, 3)
            outs.append(pd.DataFrame({"e": r[0], "r": r[1], f"{ch}_score": r[2], f"{ch}_rank": r[3]}))
        for ch in getattr(self, "extra", []):
            t0 = time.time()
            x = c[ch]
            if ch == "phon":  # exact match on the sorted Metaphone code set of the cleaned name
                r = exact_channel(ent_enc["phon_key"], rec_enc["phon_key"], x["max_postings"])
            elif ch == "addr_text":
                r = topk_cosine(ent_enc["addrtxt"], rec_enc["addrtxt"], x["k"], x["min_score"], tie_cap=c.get("tie_cap", 0))
            else:  # name_geo: name top-K' re-scored by address similarity (conjunctive name AND place rule)
                qi, di, sc, _ = topk_cosine(ent_enc["char"], rec_enc["char"], x["k_wide"], x["min_score"])
                geo = _rowdot(ent_enc["addrtxt"], rec_enc["addrtxt"], qi, di)
                d = pd.DataFrame({"e": qi, "r": di, "s": sc + x["beta"] * geo})
                d = d.sort_values(["e", "s", "r"], ascending=[True, False, True])
                d["rk"] = d.groupby("e").cumcount() + 1
                d = d[d.rk <= x["k"]]
                r = (d.e.values, d.r.values, d.s.values, d.rk.values)
            self.runtime[ch] = round(time.time() - t0, 3)
            outs.append(pd.DataFrame({"e": r[0], "r": r[1], f"{ch}_score": r[2], f"{ch}_rank": r[3]}))
        cand = outs[0]
        for o in outs[1:]:
            cand = cand.merge(o, on=["e", "r"], how="outer")
        for ch in CHANNELS:
            if f"{ch}_score" not in cand:
                cand[f"{ch}_score"] = np.nan
                cand[f"{ch}_rank"] = np.nan
            cand[f"hit_{ch}"] = cand[f"{ch}_score"].notna().astype(np.int8)
        for ch in getattr(self, "extra", []):
            cand[f"hit_{ch}"] = cand[f"{ch}_score"].notna().astype(np.int8)
        cand["n_channels"] = cand[[f"hit_{ch}" for ch in active_channels(cand)]].sum(axis=1)
        return cand.sort_values(["e", "r"]).reset_index(drop=True)


# ---------------------------------------------------------------- report (single implementation)
def _dist(counts: np.ndarray) -> dict:
    if len(counts) == 0:
        return {k: 0 for k in ("p50", "p75", "p90", "p95", "p99", "max", "mean")}
    q = np.percentile(counts, [50, 75, 90, 95, 99])
    return {"p50": round(float(q[0]), 1), "p75": round(float(q[1]), 1), "p90": round(float(q[2]), 1),
            "p95": round(float(q[3]), 1), "p99": round(float(q[4]), 1), "max": int(counts.max()), "mean": round(float(counts.mean()), 2)}


def candidate_distribution(cand: pd.DataFrame, n_entities: int, truth_pairs: set | None, mask=None) -> dict:
    """truth_pairs: set of (e, r) integer pairs restricted to the evaluated entities, or None."""
    if mask is not None:
        cand = cand[mask]
    counts = np.zeros(n_entities, int)
    vc = cand["e"].value_counts()
    counts[vc.index.values] = vc.values
    out = {"n_pairs": int(len(cand)), "entities_zero_candidates": int((counts == 0).sum()), **_dist(counts)}
    if truth_pairs is not None:
        got = set(zip(cand["e"].values, cand["r"].values)) & truth_pairs
        out["true_pairs_recovered"] = len(got)
        out["recall"] = round(len(got) / max(1, len(truth_pairs)), 4)
    else:
        out["recall"] = "UNMEASURABLE (no labels)"
    return out


def retrieval_report(cand: pd.DataFrame, n_entities: int, truth_pairs, runtime: dict, label: str,
                     out_dir="reports", name="retrieval_report", entity_filter=None) -> dict:
    base = np.ones(len(cand), bool) if entity_filter is None else cand["e"].isin(entity_filter).values
    n_eval = n_entities if entity_filter is None else len(entity_filter)
    rep = {"label": label, "n_entities_evaluated": n_eval, "n_true_pairs": None if truth_pairs is None else len(truth_pairs),
           "channels": {}}
    ents = None if entity_filter is None else np.array(sorted(entity_filter))

    def dist(mask):
        d = candidate_distribution(cand, n_entities, truth_pairs, mask)
        if ents is not None:  # recompute zero-candidate count over evaluated entities only
            have = set(cand.loc[mask, "e"].unique())
            d["entities_zero_candidates"] = int(sum(1 for e in ents if e not in have))
            counts = cand.loc[mask, "e"].value_counts().reindex(ents, fill_value=0).values
            d.update(_dist(counts))
        return d

    union_all = base & (cand["n_channels"] > 0).values
    for ch in active_channels(cand):
        m = base & (cand[f"hit_{ch}"] == 1).values
        d = dist(m)
        if truth_pairs is not None:
            others = base & ((cand["n_channels"] - cand[f"hit_{ch}"]) > 0).values
            d["unique_true_only_this_channel"] = int(
                len((set(zip(cand.loc[m, "e"], cand.loc[m, "r"])) & truth_pairs) -
                    set(zip(cand.loc[others, "e"], cand.loc[others, "r"]))))
            d["marginal_recall_over_other_channels"] = round(
                d["unique_true_only_this_channel"] / max(1, len(truth_pairs)), 4)
            d["union_p99_without_this_channel"] = dist(others)["p99"]
        d["runtime_s"] = runtime.get(ch)
        rep["channels"][ch] = d
    rep["union"] = dist(union_all)
    rep["union"]["runtime_s"] = round(sum(v for v in runtime.values() if v), 3)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    hdr = "| channel | recall | unique-only | marginal | zero-cand | P50 | P75 | P90 | P95 | P99 | max | runtime s |"
    rows = [f"# Retrieval report\n\n**{label}**\n", f"Entities evaluated: {n_eval}; true pairs: {rep['n_true_pairs']}\n",
            hdr, "|" + "---|" * 12]
    for ch, d in list(rep["channels"].items()) + [("UNION", rep["union"])]:
        rows.append(f"| {ch} | {d.get('recall')} | {d.get('unique_true_only_this_channel', '-')} | "
                    f"{d.get('marginal_recall_over_other_channels', '-')} | {d['entities_zero_candidates']} | "
                    f"{d['p50']} | {d['p75']} | {d['p90']} | {d['p95']} | {d['p99']} | {d['max']} | {d.get('runtime_s')} |")
    with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    return rep
