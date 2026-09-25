"""Address decomposition into components. Only components a source actually supplies (or that
can be parsed from its free-text address) are populated; everything else is None."""
from __future__ import annotations

import re

from src.preprocessing.normalize import basic, fold

STREET_TYPES = {"st": "street", "street": "street", "str": "strasse", "strasse": "strasse",
                "ave": "avenue", "av": "avenue", "avenue": "avenue", "avenida": "avenida",
                "rd": "road", "road": "road", "blvd": "boulevard", "bd": "boulevard",
                "boulevard": "boulevard", "rue": "rue", "rua": "rua", "r": "r", "weg": "weg",
                "allee": "allee", "marg": "marg", "ngr": "nagar", "nagar": "nagar", "tv": "travessa",
                "travessa": "travessa", "dori": "dori", "machi": "machi", "ln": "lane", "lane": "lane",
                "dr": "drive", "drive": "drive", "way": "way", "pl": "place", "place": "place"}
UNIT_MARK = {"suite", "ste", "unit", "apt", "floor", "fl", "room", "rm"}
COUNTRY_ALIASES = {
    "us": "US", "usa": "US", "united states": "US", "united states of america": "US",
    "de": "DE", "germany": "DE", "deutschland": "DE", "fr": "FR", "france": "FR",
    "in": "IN", "india": "IN", "br": "BR", "brazil": "BR", "brasil": "BR", "jp": "JP", "japan": "JP",
    "gb": "GB", "uk": "GB", "united kingdom": "GB", "es": "ES", "spain": "ES", "espana": "ES",
    "it": "IT", "italy": "IT", "italia": "IT", "ca": "CA", "canada": "CA", "mx": "MX", "mexico": "MX",
    "cn": "CN", "china": "CN", "au": "AU", "australia": "AU", "nl": "NL", "netherlands": "NL",
}
_POSTAL = re.compile(r"^(?:[a-z]?\d{5}-\d{3}|\d{3}-\d{4}|\d{5,6}|\d{4})$")


def norm_country(s) -> str | None:
    if s is None:
        return None
    k = fold(basic(s))
    return COUNTRY_ALIASES.get(k, k.upper() if k else None)


def _split_compound(tok: str) -> list[str]:
    for suf in ("strasse", "str"):
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            return [tok[: -len(suf)], suf]
    return [tok]


def parse_street(line) -> dict:
    """house number / unit / street tokens from a street line."""
    s = fold(basic(line)) if line is not None else ""
    raw = str(line or "")
    out = {"house": None, "unit": None, "street_tokens": [], "street_core": "", "numbers": []}
    if not s:
        return out
    unit = None
    m = re.search(r"#\s*(\d+)", raw)
    if m:
        unit = m.group(1)
    toks = s.split()
    street, nums = [], []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in UNIT_MARK and i + 1 < len(toks) and toks[i + 1].isdigit():
            unit = toks[i + 1]
            i += 2
            continue
        if t.isdigit():
            if t != unit:
                nums.append(t)
        else:
            for p in _split_compound(t):
                street.append(STREET_TYPES.get(p, p))
        i += 1
    out["house"] = nums[0] if nums else None
    out["unit"] = unit
    out["numbers"] = nums
    out["street_tokens"] = street
    out["street_core"] = " ".join(t for t in street if t not in STREET_TYPES.values())
    return out


def parse_full_address(s) -> dict:
    """Heuristic split of a free-text address: 'street, [postal] city, [region postal], country'."""
    if s is None or (isinstance(s, float) and s != s) or not str(s).strip():
        return {}
    segs = [x.strip() for x in str(s).split(",") if x.strip()]
    out = {"address": None, "city": None, "region": None, "postal": None, "country": None}
    if segs and fold(basic(segs[-1])) in COUNTRY_ALIASES:
        out["country"] = norm_country(segs.pop())
    street = [segs.pop(0)] if segs else []
    # "Rua X, 123" -> house number in its own segment
    if segs and re.fullmatch(r"\d{1,5}[a-zA-Z]?", segs[0]):
        street.append(segs.pop(0))
    out["address"] = ", ".join(street) if street else None
    rest = []
    for seg in segs:
        toks = fold(basic(seg)).split()
        keep = []
        for t in seg.replace(",", " ").split():
            tl = fold(t.lower())
            if _POSTAL.match(tl) and out["postal"] is None:
                out["postal"] = t
            else:
                keep.append(t)
        if not keep:
            continue
        if len(keep) == 1 and len(keep[0]) <= 3 and keep[0].isupper() and out["city"] is not None:
            out["region"] = keep[0]
        else:
            rest.append(" ".join(keep))
        del toks
    if rest:
        out["city"] = rest[0]
        if len(rest) > 1 and out["region"] is None:
            out["region"] = rest[1]
    return out


def norm_postal(p) -> str | None:
    if p is None:
        return None
    d = re.sub(r"[^0-9a-z]", "", fold(basic(p)))
    return d or None
