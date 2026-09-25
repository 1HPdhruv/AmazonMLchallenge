import numpy as np
import pandas as pd

from src.data.splits import make_splits
from src.decision.calibration import Calibrator
from src.decision.mechanisms import decide, record_exclusive
from src.retrieval.channels import AddressChannel
from src.retrieval.tfidf import HashedTfidf


def _df():
    # entity 0: two strong, one weak; entity 1: one weak; entity 2: close top-2
    return pd.DataFrame({"e": [0, 0, 0, 1, 2, 2], "r": [0, 1, 2, 3, 4, 5],
                         "s": [0.95, 0.90, 0.30, 0.40, 0.80, 0.78]})


def test_mechanisms():
    d = _df()
    assert decide(d, "threshold", {"t": 0.5}).tolist() == [1, 1, 0, 0, 1, 1]
    assert decide(d, "top1", {}).tolist() == [1, 0, 0, 1, 1, 0]
    assert decide(d, "top1_t", {"t": 0.5}).tolist() == [1, 0, 0, 0, 1, 0]
    assert decide(d, "top1_margin", {"t": 0.5, "m": 0.1}).tolist() == [0, 0, 0, 0, 0, 0]
    assert decide(d, "multi", {"t": 0.5, "delta": 0.1}).tolist() == [1, 1, 0, 0, 1, 1]
    assert decide(d, "multi_t2", {"t": 0.9, "t2": 0.25}).tolist() == [1, 1, 1, 0, 0, 0]
    assert decide(d, "gmm", {"t": 0.5}).dtype == bool


def test_record_exclusive_keeps_best_entity_per_record():
    d = pd.DataFrame({"e": [0, 1, 1], "r": [7, 7, 8], "s": [0.9, 0.6, 0.8]})
    acc = np.array([True, True, True])
    assert record_exclusive(d, acc).tolist() == [True, False, True]


def test_tfidf_idf_is_train_only_and_unseen_tokens_get_max_idf():
    m = HashedTfidf("word").fit(["acme foods", "acme motors", "zeta foods"])
    before = m.idf.copy()
    m.transform(["totally unseen tokens here"])
    assert np.array_equal(before, m.idf)                    # transform never refits
    assert m.n_fit_docs == 3
    assert m.token_idf(["qwertyzzz"])[0] == m.idf.max()     # unseen -> max IDF
    assert m.token_idf(["acme"])[0] < m.token_idf(["zeta"])[0]


def test_address_channel_rarity_floor_and_postings_cap():
    ch = AddressChannel(idf_floor=1.5, max_postings=2, k=5).fit([["p|12345"]] * 50 + [["hs|12|main"]])
    assert ch.idf("p|12345") < 1.5 < ch.idf("hs|12|main")
    q, d, s, rk = ch.retrieve([["p|12345", "hs|12|main"]], [["p|12345"], ["hs|12|main"], ["p|12345"]])
    assert d.tolist() == [1]                                 # common postal key suppressed by the floor


def test_splits_disjoint_and_deterministic():
    ids = [f"E{i}" for i in range(1000)]
    a = make_splits(ids, 7, 0.2, 0.2, 0.25)
    b = make_splits(ids, 7, 0.2, 0.2, 0.25)
    assert a.equals(b)
    assert a.value_counts().to_dict() == {"train_fit": 450, "val": 200, "test": 200, "calib": 150}


def test_calibration_monotone():
    s = np.linspace(0, 1, 200)
    y = (s + np.random.default_rng(0).normal(0, 0.2, 200) > 0.5).astype(int)
    c = Calibrator().fit(s, y).transform(s)
    assert np.all(np.diff(c) >= 0) and c.min() >= 0 and c.max() <= 1.0 + 1e-5
