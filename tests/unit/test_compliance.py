"""Static compliance guards: no network / external-lookup code, no non-competition fields, no
hard-coded country set."""
import ast
import pathlib

import numpy as np
import pandas as pd

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
FORBIDDEN_MODULES = {"requests", "urllib", "urllib3", "http", "httpx", "aiohttp", "socket", "ftplib", "smtplib",
                     "geopy", "googlemaps", "openai", "anthropic", "boto3", "webbrowser"}
FORBIDDEN_FIELDS = ("phone", "email", "website")


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0]


def test_no_network_or_external_lookup_imports():
    offenders = {(str(p.relative_to(SRC)), m) for p in SRC.rglob("*.py") for m in _imports(p) if m in FORBIDDEN_MODULES}
    assert not offenders, offenders


def test_no_non_competition_fields_in_source():
    import re
    offenders = [(str(p.relative_to(SRC)), f) for p in SRC.rglob("*.py")
                 for f in FORBIDDEN_FIELDS if re.search(rf"\b{f}(?!tic)", p.read_text(encoding="utf-8").lower())]
    assert not offenders, offenders


def test_country_is_open_set():
    from src.features.pairs import build_features  # noqa: F401  (import check)
    from src.preprocessing.address import norm_country
    assert norm_country("France") == "FR" and norm_country("Atlantis") == "ATLANTIS" and norm_country("") is None
    # nothing in src filters or one-hots on a fixed {US, India} set
    for p in SRC.rglob("*.py"):
        t = p.read_text(encoding="utf-8")
        assert "get_dummies" not in t and "OneHotEncoder" not in t, p
        assert "isin([\"US\", \"India\"])" not in t and "{\"US\", \"India\"}" not in t, p


def test_unseen_country_flows_through_representations():
    from src.preprocessing.tokenize import build_representations
    df = pd.DataFrame({"name": ["Café de Flore"], "address": ["172 Boulevard Saint-Germain"], "city": ["Paris"],
                       "region": [None], "postal": ["75006"], "country": ["France"],
                       "full_address": ["172 Boulevard Saint-Germain, 75006 Paris"]})
    rep = build_representations(df)
    assert rep.loc[0, "country_n"] == "FR" and rep.loc[0, "name_fold"] == "cafe de flore"
    assert np.all(rep["has_addr"].values)
