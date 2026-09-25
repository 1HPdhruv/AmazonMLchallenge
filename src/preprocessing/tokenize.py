"""Builds all per-row representations used by retrieval and features (additive columns)."""
from __future__ import annotations

import jellyfish
import pandas as pd

from src.preprocessing.address import norm_country, norm_postal, parse_street
from src.preprocessing.address_v2 import parse_street_v2
from src.preprocessing.legal import split_legal
from src.preprocessing.normalize import ABBREV, basic, collapse_initials, fold, legal_form, name_key, name_tokens


# Stopwords when the legal dictionary is on: 'das/da/do/die/der' are dropped from the legacy list because
# they are real name tokens (e.g. the Indian surname Das) in the competition's countries.
STOP_V2 = {"the", "and", "of", "de", "la", "le", "les", "du", "des", "et", "l", "d"}


def _legal_v2(name, cc, path):
    toks = [ABBREV.get(t, t) for t in fold(basic(name)).split()]
    core, classes = split_legal(toks, cc, path)
    clean = [t for t in core if t not in STOP_V2] or core
    return " ".join(core), " ".join(clean), " ".join(sorted(set(clean))), "+".join(classes)


def build_representations(df: pd.DataFrame, norm: dict | None = None) -> pd.DataFrame:
    norm = norm or {}
    if norm.get("collapse_initials"):
        df = df.copy()
        for c in ("name", "address", "full_address"):
            df[c] = df[c].map(collapse_initials)
    out = pd.DataFrame(index=df.index)
    out["name_raw"] = df["name"]
    out["name_norm"] = df["name"].map(basic)                       # accents kept
    out["name_fold"] = out["name_norm"].map(fold)                  # accent-folded (additive)
    if norm.get("legal_dictionary"):
        cc = df["country"].map(norm_country)
        v2 = [_legal_v2(n, c, norm["legal_dictionary"]) for n, c in zip(df["name"], cc)]
        out["name_fold_core"] = [x[0] for x in v2]                 # legal span removed (char channel)
        out["name_clean"] = [x[1] for x in v2]
        out["name_key"] = [x[2] for x in v2]
        out["legal"] = [x[3] for x in v2]                          # canonical legal class(es)
    else:
        toks = df["name"].map(name_tokens)
        out["name_clean"] = toks.map(" ".join)
        out["name_key"] = df["name"].map(name_key)
        out["legal"] = df["name"].map(legal_form)
        out["name_fold_core"] = out["name_fold"]
    if norm.get("address_parser") == "v2":
        ps = pd.Series([parse_street_v2(a, c) for a, c in zip(df["address"], df["country"])], index=df.index)
    else:
        ps = df["address"].map(parse_street)
    out["house"] = ps.map(lambda d: d["house"])
    out["unit"] = ps.map(lambda d: d["unit"])
    out["street_core"] = ps.map(lambda d: d["street_core"])
    out["street_norm"] = ps.map(lambda d: " ".join(d["street_tokens"]))
    out["numbers"] = ps.map(lambda d: d["numbers"])
    out["city_n"] = df["city"].map(lambda s: fold(basic(s)) or None)
    out["region_n"] = df["region"].map(lambda s: fold(basic(s)) or None)
    out["postal_n"] = df["postal"].map(norm_postal)
    out["country_n"] = df["country"].map(norm_country)               # open set: unknown labels pass through
    # full free-text address (competition field business_address); components above are parsed from it
    out["addr_fold"] = df["full_address"].map(lambda s: fold(basic(s)))
    out["has_name"] = out["name_clean"].str.len() > 0
    out["has_addr"] = df["full_address"].map(ok)
    # phonetic codes of the cleaned name tokens (Metaphone + NYSIIS; jellyfish, MIT) for transliteration/typos
    mp = out["name_clean"].map(lambda x: [jellyfish.metaphone(t) for t in x.split()] if x else [])
    ny = out["name_clean"].map(lambda x: [jellyfish.nysiis(t) for t in x.split()] if x else [])
    out["name_mp"] = mp.map(" ".join)
    out["name_ny"] = ny.map(" ".join)
    out["name_mp_key"] = mp.map(lambda c: " ".join(sorted(set(x for x in c if x))))
    # NOTE: pandas 3 stores missing strings as NaN; consumers use `ok()` rather than `is None`.
    return out


def ok(v) -> bool:
    """True for a usable (non-empty) string value; False for None / NaN / ''."""
    return isinstance(v, str) and v != ""
