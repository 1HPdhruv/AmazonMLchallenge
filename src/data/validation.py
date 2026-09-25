"""Schema + data validation. Critical violations raise; everything else is reported."""
from __future__ import annotations

import json
import os

import pandas as pd


class SchemaError(Exception):
    pass


def _field_stats(df: pd.DataFrame) -> dict:
    return {c: {"non_null_frac": round(float(df[c].notna().mean()), 4),
                "n_unique": int(df[c].nunique(dropna=True))} for c in df.columns}


def validate(entities: pd.DataFrame, records: pd.DataFrame, labels, present_fields: dict,
             mode_label: str, adapter_issues=()) -> dict:
    critical, warnings = [], list(adapter_issues)
    if entities["entity_id"].isna().any():
        critical.append("null entity_id in source1")
    if entities["entity_id"].duplicated().any():
        critical.append(f"{int(entities['entity_id'].duplicated().sum())} duplicate entity_id in source1")
    if records["record_id"].isna().any():
        critical.append("null record_id in records")
    if records["record_id"].duplicated().any():
        critical.append(f"{int(records['record_id'].duplicated().sum())} duplicate record_id across sources")
    for name, df in (("source1", entities), ("records", records)):
        both = df["name"].isna() & df["full_address"].isna()
        if both.any():
            warnings.append(f"{name}: {int(both.sum())} rows with neither name nor address")
    lab = {}
    if labels is not None:
        ue = set(entities["entity_id"])
        ur = set(records["record_id"])
        bad_e = ~labels["entity_id"].isin(ue)
        bad_r = ~labels["record_id"].isin(ur)
        if bad_e.any() or bad_r.any():
            critical.append(f"labels reference unknown ids: {int(bad_e.sum())} entity, {int(bad_r.sum())} record")
        if labels.duplicated().any():
            warnings.append(f"{int(labels.duplicated().sum())} duplicate label rows (deduplicated)")
        per_rec = labels.groupby("record_id")["entity_id"].nunique()
        n_matches = labels.groupby("entity_id")["record_id"].nunique().reindex(entities["entity_id"]).fillna(0)
        lab = {"n_label_pairs": int(len(labels)),
               "records_linked_to_multiple_entities": int((per_rec > 1).sum()),
               "entities_no_match": int((n_matches == 0).sum()),
               "entities_single_match": int((n_matches == 1).sum()),
               "entities_multi_match": int((n_matches > 1).sum()),
               "max_matches_per_entity": int(n_matches.max())}
    rep = {"label": mode_label,
           "n_entities": int(len(entities)),
           "n_records": int(len(records)),
           "records_per_source": records["source"].value_counts().to_dict(),
           "present_fields_per_source": present_fields,
           "countries_source1": entities["country"].fillna("<empty>").value_counts().to_dict(),
           "countries_records": records["country"].fillna("<empty>").value_counts().to_dict(),
           "entity_field_stats": _field_stats(entities),
           "record_field_stats": _field_stats(records),
           "labels": lab, "critical": critical, "warnings": warnings}
    if critical:
        raise SchemaError("; ".join(critical))
    return rep


def write_schema_report(rep: dict, out_dir: str = "reports"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "schema_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
    lines = [f"# Schema report\n", f"**{rep['label']}**\n",
             f"- entities (Source 1): {rep['n_entities']}", f"- records: {rep['n_records']} {rep['records_per_source']}",
             "", "## Fields present per source"]
    for k, v in rep["present_fields_per_source"].items():
        lines.append(f"- {k}: {', '.join(v)}")
    lines += ["", f"## Countries (open set)", f"- Source 1: {rep['countries_source1']}",
              f"- records: {rep['countries_records']}", "", "## Field completeness (Source 1)", "| field | non-null | unique |", "|---|---|---|"]
    lines += [f"| {k} | {v['non_null_frac']} | {v['n_unique']} |" for k, v in rep["entity_field_stats"].items()]
    lines += ["", "## Field completeness (records)", "| field | non-null | unique |", "|---|---|---|"]
    lines += [f"| {k} | {v['non_null_frac']} | {v['n_unique']} |" for k, v in rep["record_field_stats"].items()]
    lines += ["", "## Labels", *[f"- {k}: {v}" for k, v in rep["labels"].items()],
              "", "## Warnings", *([f"- {w}" for w in rep["warnings"]] or ["- none"])]
    with open(os.path.join(out_dir, "schema_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
