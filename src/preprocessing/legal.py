"""Country-aware legal-form handling driven by configs/legal_forms.yaml (a data file, not code).

Only trailing spans are stripped (at most two, never the first token), so a distinguishing leading
token such as 'AB' in 'AB Enterprises' or 'Das' in 'Das Brothers & Co' is never deleted. Countries
without an entry fall back to `_generic`, which holds only unambiguous forms.
"""
from __future__ import annotations

import functools
import os

import yaml

DEFAULT_PATH = os.path.join("configs", "legal_forms.yaml")


@functools.lru_cache(maxsize=4)
def load(path: str = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out = {}
    for cc, forms in raw.items():
        out[cc] = {tuple(k.split()): v for k, v in forms.items()}
    return out


def split_legal(tokens: list[str], country_code, path: str = DEFAULT_PATH):
    """tokens: folded name tokens. Returns (core_tokens, legal_classes) — classes in name order."""
    table = load(path)
    forms = table.get(country_code) if isinstance(country_code, str) and country_code in table else table["_generic"]
    max_len = max(len(k) for k in forms)
    core, classes = list(tokens), []
    for _ in range(2):
        hit = None
        for n in range(min(max_len, len(core) - 1), 0, -1):   # len(core)-1: never strip the first token
            span = tuple(core[-n:])
            if span in forms:
                hit = (n, forms[span])
                break
        if not hit:
            break
        core = core[:-hit[0]]
        classes.insert(0, hit[1])
    return core, classes
