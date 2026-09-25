"""Conservative, additive text normalization. Raw text is always kept; every representation
here is an additional column, never a replacement."""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

# Legal-form tokens (static linguistic list, not learned). Stripped only for keys/retrieval;
# the legal form is still compared as its own feature.
LEGAL = {"inc", "incorporated", "llc", "corp", "corporation", "co", "ltd", "limited", "gmbh", "ag",
         "kg", "sarl", "sa", "sas", "pvt", "private", "ltda", "me", "kk", "plc", "llp", "bv", "nv",
         "srl", "spa", "oy", "ab", "as", "pte", "pty", "sl", "lp", "company"}
STOP = {"the", "and", "of", "de", "la", "le", "les", "das", "der", "die", "du", "des", "da", "do", "et"}
ABBREV = {"intl": "international", "int'l": "international", "svcs": "services", "svc": "services",
          "hldgs": "holdings", "hldg": "holdings", "ent": "enterprises", "eng": "engineering",
          "elec": "electronics", "constr": "construction", "bros": "brothers", "mfg": "manufacturing",
          "grp": "group", "tech": "technology", "natl": "national", "assoc": "associates"}


def basic(s) -> str:
    """NFKC + casefold + punctuation->space + whitespace collapse. Accents kept."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold().replace("&", " and ")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def fold(s: str) -> str:
    """Accent folding (additive representation). ß -> ss handled explicitly."""
    s = s.replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def name_tokens(s) -> list[str]:
    """Folded tokens with abbreviations expanded, legal forms and stopwords removed."""
    toks = [ABBREV.get(t, t) for t in fold(basic(s)).split()]
    out = [t for t in toks if t not in LEGAL and t not in STOP]
    return out if out else [t for t in toks if t not in STOP]


def legal_form(s) -> str:
    toks = fold(basic(s)).split()
    return " ".join(t for t in toks if t in LEGAL)


def name_key(s) -> str:
    """Order-insensitive exact-match key."""
    return " ".join(sorted(set(name_tokens(s))))


def digits(s) -> str:
    return re.sub(r"\D", "", s or "")


# two or more single letters, each dotted except possibly the last: S.K. / M. G. / L.L.C. / A.P.J.
_INITIALS = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\.\s*)+[A-Za-z]\.?(?![A-Za-z])")
_HONORIFIC = re.compile(r"^\s*(?:m\s*/\s*s\.?|messrs\.?)\s+", re.I)


def collapse_initials(s):
    """Raw-text fix for dotted initialisms before punctuation removal: 'S.K. Das' -> 'SK Das',
    'M.G. Road' -> 'MG Road', 'L.L.C.' -> 'LLC'; strips the Indian business honorific 'M/s'/'Messrs'."""
    if not isinstance(s, str) or not s:
        return s
    s = _HONORIFIC.sub("", s)
    return _INITIALS.sub(lambda m: re.sub(r"[.\s]", "", m.group(0)) + " ", s).strip()
