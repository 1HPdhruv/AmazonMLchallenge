import numpy as np

from src.evaluation.entity_metrics import fbeta, macro_f05, paired_bootstrap, per_entity


def test_fbeta_edge_cases():
    assert fbeta(set(), set()) == 1.0            # correct no-match
    assert fbeta({"r1"}, set()) == 0.0           # no-match false positive: no partial credit
    assert fbeta(set(), {"r1"}) == 0.0           # missed match
    assert fbeta({"r1"}, {"r1"}) == 1.0
    p, r = 0.5, 1.0                              # {r1,r2} vs {r1}
    assert abs(fbeta({"r1", "r2"}, {"r1"}) - 1.25 * p * r / (0.25 * p + r)) < 1e-12


def test_vectorized_matches_set_based():
    rng = np.random.default_rng(0)
    n_e, rows = 50, 400
    e = rng.integers(0, n_e, rows)
    r = np.arange(rows)
    y = (rng.random(rows) < 0.2).astype(np.int8)
    acc = rng.random(rows) < 0.3
    extra_true = rng.integers(0, 2, n_e)          # true matches never retrieved
    n_true = np.bincount(e, weights=y, minlength=n_e).astype(int) + extra_true
    ents = np.arange(n_e)
    pe = per_entity(e, acc, y, ents, n_true, np.bincount(e, minlength=n_e))
    for i in ents:
        t = {j for j in r[(e == i) & (y == 1)]} | {f"miss{k}" for k in range(extra_true[i])}
        p = {j for j in r[(e == i) & acc]}
        assert abs(pe.f05[i] - fbeta(p, t)) < 1e-9
    assert abs(macro_f05(e, acc, y, ents, n_true) - pe.f05.mean()) < 1e-12


def test_paired_bootstrap_sign():
    a = np.zeros(200)
    b = np.ones(200)
    assert paired_bootstrap(a, b)["p_improve"] == 1.0
    assert paired_bootstrap(b, a)["p_improve"] == 0.0
