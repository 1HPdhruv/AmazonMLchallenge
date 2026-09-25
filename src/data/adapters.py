"""Data adapters.

CompetitionDataAdapter reads the OFFICIAL layout (configs/competition_facts.md):
    <root>/dataset/{train,test}/{split}_source{1,2,3}.tsv   columns: entity_id, business_name,
                                                             business_address, country
    <root>/dataset/train/train_ground_truth.tsv              columns: source1_entity_id, matched_entity_ids
It uses ONLY those four source fields. Source identity comes from the file (and is checked
against the S1-/S2-/S3- id prefix). Country is kept as an open-set string. Address components
(street / city / region / postal) are PARSED from business_address, because no other address
field exists. Nothing in this class encodes synthetic distributions.

SyntheticDataAdapter is the same reader pointed at a generated synthetic tree in the identical
layout. It can generate that tree and read the synthetic-only hidden test labels (used solely to
score output mechanics; never available for real data).

Canonical tables returned by load(split):
  entities: entity_id, name, address(street part), city, region, postal, country, full_address
  records : record_id, source, name, address, city, region, postal, country, full_address
  labels  : entity_id, record_id (train only; None for test)
"""
from __future__ import annotations

import csv
import logging
import os

import pandas as pd

from src.preprocessing.address import parse_full_address
from src.preprocessing.address_v2 import parse_full_address_v2

log = logging.getLogger("er")
SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]
PREFIX = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}


class SchemaViolation(Exception):
    pass


def read_tsv(path: str, expected_cols: list[str]) -> pd.DataFrame:
    """Tab-separated reader; every value is a string, empty cells become ''.
    Quotes are LITERAL (QUOTE_NONE): the official files are unquoted, and CSV-style quote handling
    can silently merge a field starting with '"' into its neighbour while keeping the row count
    unchanged (caught by tests). A line with the wrong number of fields fails loudly. Fields that
    look CSV-quoted are reported so a human can inspect them."""
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\r\n") for line in f]
    lines = [ln for ln in lines if ln != ""]
    n_lines = len(lines) - 1
    n_fields = lines[0].count("\t") + 1 if lines else 0
    too_many = [i + 2 for i, ln in enumerate(lines[1:]) if ln.count("\t") + 1 > n_fields]
    if too_many:  # pandas would silently turn the first column into an index here
        raise SchemaViolation(f"{path}: {len(too_many)} lines have more than {n_fields} fields, e.g. line {too_many[0]}")
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[], encoding="utf-8",
                         quoting=csv.QUOTE_NONE, index_col=False)
    except pd.errors.ParserError as ex:
        raise SchemaViolation(f"{path}: malformed TSV ({ex})") from ex
    if len(df) != n_lines:
        raise SchemaViolation(f"{path}: parsed {len(df)} rows but file has {n_lines} data lines")
    short = int(df.isna().any(axis=1).sum())   # e.g. "S1-00003" with no trailing tab (official example style)
    if short:
        log.warning("%s: %d rows with missing trailing fields treated as empty", path, short)
        df = df.fillna("")
    quoted = int(sum(df[c].astype(str).str.match(r'^".*"$').sum() for c in df.columns))
    if quoted:
        log.warning("%s: %d fields look CSV-quoted; kept literally. Inspect before trusting names.", path, quoted)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if list(df.columns) != expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise SchemaViolation(f"{path}: missing columns {missing}; found {list(df.columns)}")
        log.warning("%s: extra/reordered columns %s (only %s are used)", path, list(df.columns), expected_cols)
    for c in expected_cols:
        df[c] = df[c].astype(str).str.strip()
    return df[expected_cols]


def _canonical(df: pd.DataFrame, id_col: str, parser: str = "v1") -> pd.DataFrame:
    if parser == "v2":
        parsed = pd.Series([parse_full_address_v2(s, c) if s else {} for s, c in zip(df["business_address"], df["country"])],
                           index=df.index)
    else:
        parsed = df["business_address"].map(lambda s: parse_full_address(s) if s else {})
    out = pd.DataFrame({
        id_col: df["entity_id"].values,
        "name": df["business_name"].where(df["business_name"] != "", None).values,
        "address": parsed.map(lambda d: d.get("address")).values,
        "city": parsed.map(lambda d: d.get("city")).values,
        "region": parsed.map(lambda d: d.get("region")).values,
        "postal": parsed.map(lambda d: d.get("postal")).values,
        "country": df["country"].where(df["country"] != "", None).values,   # open set, raw string
        "full_address": df["business_address"].where(df["business_address"] != "", None).values,
    })
    return out.astype(object).where(out.notna(), None)


class CompetitionDataAdapter:
    def __init__(self, root: str, parser: str = "v1"):
        self.root = root
        self.parser = parser
        self.present_fields = {k: list(SOURCE_COLS) + ["(parsed from business_address: street/city/region/postal)"]
                               for k in PREFIX}
        self.issues: list[str] = []

    def path(self, split: str, name: str) -> str:
        return os.path.join(self.root, "dataset", split, f"{split}_{name}.tsv" if name.startswith("source")
                            else f"{name}.tsv")

    def available(self, split: str) -> bool:
        return all(os.path.exists(self.path(split, s)) for s in PREFIX)

    def load(self, split: str):
        raw = {}
        for s, pre in PREFIX.items():
            df = read_tsv(self.path(split, s), SOURCE_COLS)
            bad = ~df["entity_id"].str.startswith(pre)
            if bad.any():
                raise SchemaViolation(f"{split}_{s}: {int(bad.sum())} ids without prefix {pre}, e.g. "
                                      f"{df.loc[bad, 'entity_id'].head(3).tolist()}")
            raw[s] = df
        ents = _canonical(raw["source1"], "entity_id", self.parser)
        recs = pd.concat([_canonical(raw[s], "record_id", self.parser).assign(source=s) for s in ("source2", "source3")],
                         ignore_index=True)
        recs = recs[["record_id", "source"] + [c for c in recs.columns if c not in ("record_id", "source")]]
        labels = self.load_labels(split, set(ents["entity_id"]), set(recs["record_id"])) if split == "train" else None
        return ents, recs, labels

    def load_labels(self, split, ent_ids: set, rec_ids: set, path: str | None = None):
        path = path or os.path.join(self.root, "dataset", "train", "train_ground_truth.tsv")
        if not os.path.exists(path):
            self.issues.append(f"ground truth not found: {path}")
            return None
        gt = read_tsv(path, GT_COLS)
        rows = [(e, m.strip()) for e, ms in zip(gt.source1_entity_id, gt.matched_entity_ids)
                for m in ms.split(",") if m.strip()]
        labels = pd.DataFrame(rows, columns=["entity_id", "record_id"]).drop_duplicates()
        missing_e = ent_ids - set(gt.source1_entity_id)
        if missing_e:
            self.issues.append(f"{len(missing_e)} Source-1 entities absent from ground truth (treated as no-match)")
        if gt.source1_entity_id.duplicated().any():
            self.issues.append(f"{int(gt.source1_entity_id.duplicated().sum())} duplicate rows in ground truth")
        bad = ~labels.record_id.str[:3].isin(["S2-", "S3-"])
        if bad.any():
            raise SchemaViolation(f"ground truth references non S2-/S3- ids: {labels.record_id[bad].head(3).tolist()}")
        unknown = ~labels.record_id.isin(rec_ids) | ~labels.entity_id.isin(ent_ids)
        if unknown.any():
            raise SchemaViolation(f"ground truth references {int(unknown.sum())} ids not in the {split} files")
        return labels


class SyntheticDataAdapter(CompetitionDataAdapter):
    """Synthetic tree in the official layout. SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE."""

    def __init__(self, root: str, n_entities: int = 6000, seed: int = 7, parser: str = "v1"):
        super().__init__(root, parser)
        if not self.available("train"):
            from tests.synthetic.generate_data import generate
            generate(n_entities, seed, root)

    def hidden_test_labels(self, ent_ids: set, rec_ids: set):
        return self.load_labels("test", ent_ids, rec_ids, os.path.join(self.root, "_hidden", "test_ground_truth.tsv"))


def make_adapter(base: dict) -> CompetitionDataAdapter:
    d = base["data"]
    parser = (base.get("normalization") or {}).get("address_parser", "v1")
    if base["mode"] == "synthetic":
        return SyntheticDataAdapter(d["root"], base["synthetic"]["n_entities"], base["synthetic"]["seed"], parser)
    return CompetitionDataAdapter(d["root"], parser)
