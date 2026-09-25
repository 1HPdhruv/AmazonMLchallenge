"""Address parser v2 (audit findings #5, #9, #10): content-based component assignment.

Enabled by configs/base.yaml `normalization.address_parser: v2`. The country hint is the row's own
country label (open set; unknown countries fall back to country-agnostic patterns).
Fixes, each covered by tests/unit/test_address_v2.py:
  * landmark segments ('Near SBI ATM', 'Opp. Bus Stand') are set aside instead of becoming the street/city
  * a comma-separated unit segment (', Suite 120') becomes the unit instead of the city
  * the street segment is chosen by content (digit + word or a street-type word), not by position
  * postal codes are found anywhere (ZIP+4, spaced/glued Indian PIN, 'F-75008'), rightmost match wins
  * a trailing 2-letter segment is only a country when it is the row's own code ('CA'/'IN' states survive)
  * French articles / house suffixes / Indian door markers are removed from the street core
"""
from __future__ import annotations

import re

from src.preprocessing.address import COUNTRY_ALIASES, STREET_TYPES, UNIT_MARK, _split_compound, norm_country
from src.preprocessing.normalize import basic, fold

STREET_TYPES_V2 = dict(STREET_TYPES)
STREET_TYPES_V2.update({
    "chemin": "chemin", "ch": "chemin", "chem": "chemin", "impasse": "impasse", "imp": "impasse",
    "quai": "quai", "cours": "cours", "crs": "cours", "square": "square", "sq": "square",
    "route": "route", "rte": "route", "faubourg": "faubourg", "fbg": "faubourg",
    "salai": "salai", "gali": "gali",
})
STREET_STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "del", "della", "von", "van", "der", "the", "of",
               "bis", "ter", "quater", "no", "plot", "door", "h", "shop", "flat"}
_LANDMARK = re.compile(r"^\s*(near|nr|opp|opposite|behind|beside|next to|adjacent to|above|below|"
                       r"in front of|landmark|close to)\b", re.I)
# unit value is short (<= 4 chars): 'FL 15045' is a state + ZIP, not 'floor 15045'
_UNIT_SEG = re.compile(r"^\s*(suite|ste|unit|apt|apartment|fl|floor|room|rm|#|shop|office|flat)\s*\.?\s*[\w-]{1,4}\s*$", re.I)
_DOOR_ONLY = re.compile(r"(?i)^\s*(?:#|no\.?|door no\.?|plot(?: no\.?)?|h\.?\s?no\.?)?\s*[\dA-Za-z]*\d[\dA-Za-z/-]*\s*$")
_POSTAL_BY_CC = {
    "US": [re.compile(r"(?<![\d-])(\d{5})(?:-\d{4})?(?!\d)")],
    "IN": [re.compile(r"(?<!\d)([1-9]\d{2})\s?(\d{3})(?!\d)")],
    "FR": [re.compile(r"(?<!\d)(\d{5})(?!\d)")],
}
_POSTAL_ANY = [re.compile(r"(?<!\d)(\d{5})-(\d{3})(?!\d)"), re.compile(r"(?<!\d)(\d{3})-(\d{4})(?!\d)"),
               re.compile(r"(?<![\d-])(\d{5})(?:-\d{4})?(?!\d)"), re.compile(r"(?<!\d)([1-9]\d{2})\s?(\d{3})(?!\d)")]


def _find_postal(seg: str, cc):
    for rx in _POSTAL_BY_CC.get(cc, _POSTAL_ANY):
        ms = list(rx.finditer(seg))
        if ms:
            m = ms[-1]
            return "".join(g for g in m.groups() if g), m.span()
    return None, None


_REGION_POSTAL = re.compile(r"(?i)^\s*[a-z]{2,3}\s*-?\s*\d{3}\s?\d{2,3}(?:-\d{3,4})?\s*$")


def _has_street(seg: str) -> bool:
    if _REGION_POSTAL.match(seg):          # 'FL 15045' / 'MH 400001' is region + postal, not a street
        return False
    t = fold(basic(seg)).split()
    return (any(x.isdigit() for x in t) and any(x.isalpha() and len(x) > 1 for x in t)) or \
        any(x in STREET_TYPES_V2 for x in t)


def _is_region(seg: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{2,3}", seg.strip())) and seg.strip().isupper()


def parse_full_address_v2(s, country_hint=None) -> dict:
    if s is None or (isinstance(s, float) and s != s) or not str(s).strip():
        return {}
    cc = norm_country(country_hint) if isinstance(country_hint, str) and country_hint else None
    segs = [x.strip() for x in str(s).split(",") if x.strip()]
    out = {"address": None, "city": None, "region": None, "postal": None, "country": None,
           "landmark": None, "unit": None}
    if segs:  # trailing country: spelled-out names, or the row's own code (so 'CA'/'IN' US states survive)
        k = fold(basic(segs[-1]))
        if k in COUNTRY_ALIASES and (len(k) > 3 or COUNTRY_ALIASES[k] == cc):
            out["country"] = COUNTRY_ALIASES[k]
            segs.pop()
    kept = []
    for seg in segs:
        if _LANDMARK.match(seg):
            out["landmark"] = seg if out["landmark"] is None else out["landmark"] + ", " + seg
        elif _UNIT_SEG.match(seg) and not (cc == "IN" and seg.strip().startswith("#")):
            out["unit"] = seg
        else:
            kept.append(seg)
    segs = kept
    if not segs:
        return out
    i = next((j for j, seg in enumerate(segs) if _has_street(seg)), 0)
    if i > 0 and _DOOR_ONLY.match(segs[i - 1]):   # '#12-3-456, Road No. 5' -> door number belongs to the street
        i -= 1
    street = [segs[i]]
    if _DOOR_ONLY.match(segs[i]) and i + 1 < len(segs):
        street.append(segs[i + 1])
        del segs[i:i + 2]
    else:
        del segs[i]
        if i < len(segs) and re.fullmatch(r"\d{1,5}[a-zA-Z]?", segs[i]):
            street.append(segs.pop(i))
    out["address"] = ", ".join(street)
    pidx = None
    for j in range(len(segs) - 1, -1, -1):
        pc, span = _find_postal(segs[j], cc)
        if pc:
            out["postal"], pidx = pc, j
            segs[j] = (segs[j][:span[0]] + " " + segs[j][span[1]:]).strip(" -")
            if segs[j].upper().startswith("F-"):
                segs[j] = segs[j][2:].strip()
            break
    if out["postal"] is None:  # comma-less address: postal inside the street text, never its first number
        line = out["address"]
        pc, span = _find_postal(line, cc)
        first_num = re.search(r"\d+", line)
        if pc and first_num and span[0] > first_num.start():
            out["postal"] = pc
            out["address"] = (line[:span[0]] + " " + line[span[1]:]).strip(" ,-")
    rest = []
    for j, seg in enumerate(segs):
        seg = seg.strip(" -")
        if not seg:
            continue
        if _is_region(seg):
            out["region"] = seg
            continue
        toks = seg.split()
        if len(toks) > 1 and _is_region(toks[-1]) and j == pidx:   # 'NY' left over from 'NY 10001'
            out["region"] = toks[-1]
            seg = " ".join(toks[:-1])
        rest.append((j, seg))
    if rest:
        if pidx is not None and any(j == pidx for j, _ in rest):
            city = next(seg for j, seg in rest if j == pidx)
        elif pidx is not None and any(j < pidx for j, _ in rest):
            city = [seg for j, seg in rest if j < pidx][-1]
        else:
            city = rest[-1][1]
        out["city"] = city
        others = [seg for _, seg in rest if seg != city]
        if others and out["region"] is None:
            out["region"] = others[-1]
    return out


def parse_street_v2(line, country_hint=None) -> dict:
    cc = norm_country(country_hint) if isinstance(country_hint, str) and country_hint else None
    raw = "" if line is None or (isinstance(line, float) and line != line) else str(line)
    raw = re.sub(r"(\d)([A-Za-z]{3,})", r"\1 \2", raw)             # '257pine' -> '257 pine'
    raw = re.sub(r"(?i)(\d)\s*(bis|ter|quater)\b", r"\1", raw)     # French house-number suffixes
    s = fold(basic(raw)) if raw else ""
    out = {"house": None, "unit": None, "street_tokens": [], "street_core": "", "numbers": []}
    if not s:
        return out
    unit = None
    m = re.search(r"#\s*(\d+)", raw)
    if m and cc != "IN":   # Indian '#12-3-456' is a door number, not a US-style unit
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
                street.append(STREET_TYPES_V2.get(p, p))
        i += 1
    types = set(STREET_TYPES_V2.values())
    out["house"] = nums[0] if nums else None
    out["unit"] = unit
    out["numbers"] = nums
    out["street_tokens"] = [t for t in street if t not in STREET_STOP]
    out["street_core"] = " ".join(t for t in street if t not in types and t not in STREET_STOP)
    return out
