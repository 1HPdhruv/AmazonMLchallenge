"""Unit tests for the deep-improvement pass components (parser v2, legal dictionary, initials, decision)."""
import numpy as np
import pandas as pd

from src.decision.mechanisms import decide, record_exclusive
from src.preprocessing.address_v2 import parse_full_address_v2 as P, parse_street_v2 as S
from src.preprocessing.legal import split_legal
from src.preprocessing.normalize import collapse_initials
from src.retrieval.tfidf import HashedTfidf, topk_cosine


def test_parser_v2_landmark_unit_postal_region():
    d = P("Near SBI ATM, 12 MG Road, Mumbai - 400001, MH", "India")
    assert (d["address"], d["city"], d["region"], d["postal"], d["landmark"]) == \
        ("12 MG Road", "Mumbai", "MH", "400001", "Near SBI ATM")
    d = P("12 Main St, Suite 120, New York, NY 10001", "US")
    assert (d["city"], d["region"], d["postal"], d["unit"]) == ("New York", "NY", "10001", "Suite 120")
    assert P("123 Main St, Los Angeles, CA", "US")["region"] == "CA"          # CA state is not Canada here
    assert P("12 Oak St, Springfield, IL 62701-1234", "US")["postal"] == "62701"   # ZIP+4
    assert P("5 Park Rd, Bangalore-560001", "India")["postal"] == "560001"          # glued PIN
    assert P("No. 5/2, MG Road, Indiranagar, Bengaluru - 560 038", "India")["postal"] == "560038"  # spaced PIN
    assert P("12 Rue de Oak, F-75002 Paris, France", "France")[["city", "postal"][0]] == "Paris"
    assert P("MIAMI, FL 15045, 2243 FLEURS STREET", "US")["address"] == "2243 FLEURS STREET"  # reordered


def test_street_v2_articles_suffixes_doors():
    assert S("12 bis Rue de la Paix", "France")["street_core"] == "paix"
    assert S("12 Rue de Oak", "France")["house"] == "12"
    assert S("#12-3-456, Road No. 5", "India")["house"] == "12"        # Indian door number, not a unit
    assert S("12 Main St #40", "US")["unit"] == "40"
    assert S("257pine street", "US")["house"] == "257"


def test_legal_dictionary_trailing_only_and_generic_fallback():
    assert split_legal(["ab", "enterprises"], "US") == (["ab", "enterprises"], [])
    assert split_legal(["das", "brothers", "and", "co"], "IN") == (["das", "brothers"], ["AND_CO"])
    assert split_legal(["sharma", "traders", "private", "limited"], "IN")[1] == ["PVT_LTD"]
    assert split_legal(["sharma", "traders", "pvt", "ltd"], "IN")[1] == ["PVT_LTD"]
    assert split_legal(["acme", "sa"], "ATLANTIS") == (["acme", "sa"], [])      # ambiguous form kept for unseen country
    assert split_legal(["acme", "gmbh"], "ATLANTIS") == (["acme"], ["GMBH"])
    assert split_legal(["sarl"], "FR") == (["sarl"], [])                        # never strips the only token


def test_collapse_initials():
    assert collapse_initials("S.K. Das Pvt Ltd").split() == ["SK", "Das", "Pvt", "Ltd"]
    assert collapse_initials("M/s Sharma Traders") == "Sharma Traders"
    assert collapse_initials("Acme L.L.C.") == "Acme LLC"
    assert collapse_initials("J. Smith & Co") == "J. Smith & Co"                # single initial untouched
    assert collapse_initials("St. Louis Bread Co") == "St. Louis Bread Co"


def test_record_exclusive_all_mode_uses_all_candidates_and_drops_ties():
    d = pd.DataFrame({"e": [0, 1, 2, 3], "r": [7, 7, 8, 8], "s": [0.6, 0.9, 0.8, 0.8]})
    acc = np.array([True, False, True, True])
    # record 7's best entity (1) was NOT accepted -> entity 0 must not keep it; record 8 is an exact tie -> dropped
    assert record_exclusive(d, acc, "all").tolist() == [False, False, False, False]
    assert record_exclusive(d, acc, "accepted").tolist() == [True, False, True, True]


def test_abs2_allows_secondary_bar_above_top1_bar():
    d = pd.DataFrame({"e": [0, 0, 0], "r": [1, 2, 3], "s": [0.95, 0.9, 0.5]})
    assert decide(d, "abs2", {"t1": 0.4, "t2": 0.85}).tolist() == [True, True, False]


def test_unseen_idf_median_and_tie_block():
    m = HashedTfidf("word", unseen="median").fit(["acme foods", "acme motors", "zeta foods", "beta labs"])
    assert m.token_idf(["qwertyzzz"])[0] == np.median(m.idf[m.seen])
    assert m.token_idf(["qwertyzzz"])[0] < np.log(1 + 4) + 1                    # below the df=0 'max' IDF
    assert m.n_unseen(["qwertyzzz", "acme"]) == 1
    import scipy.sparse as sp
    Q = sp.csr_matrix(np.array([[1.0, 0.0]]))
    D = sp.csr_matrix(np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]]))
    assert len(topk_cosine(Q, D, 2, 0.1)[1]) == 2
    assert len(topk_cosine(Q, D, 2, 0.1, tie_cap=50)[1]) == 5                   # whole tied block kept
