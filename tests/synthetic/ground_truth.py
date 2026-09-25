"""Ground-truth helpers for the synthetic benchmark. SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE."""
from __future__ import annotations

import pandas as pd


def truth_sets(labels: pd.DataFrame, entity_ids) -> dict:
    """entity_id -> frozenset(record_id); entities without labels map to the empty set."""
    g = labels.groupby("entity_id")["record_id"].apply(frozenset).to_dict()
    return {e: g.get(e, frozenset()) for e in entity_ids}


def match_type_counts(truth: dict) -> dict:
    n = pd.Series([len(v) for v in truth.values()])
    return {"no_match": int((n == 0).sum()), "single": int((n == 1).sum()), "multi": int((n > 1).sum())}
