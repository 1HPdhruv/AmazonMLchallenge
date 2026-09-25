"""Integration: data -> retrieval -> features -> model -> decision, plus leakage / compliance guards.
SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE (mechanics only; no performance claims)."""
import numpy as np
import pytest

from src.data.validation import SchemaError, validate
from src.pipeline.core import Context, load_cfg, reference_eval, train_model

FORBIDDEN_FEATURE_TOKENS = ("phone", "email", "website", "domain", "url", "registr")


@pytest.fixture(scope="module")
def ctx(small_cfg):
    return Context().prepare(small_cfg, load_cfg("model"))


def test_train_only_statistics(ctx):
    n_fit = len(ctx.idx["train_fit"])
    assert ctx.cg.char.n_fit_docs == n_fit and ctx.cg.word.n_fit_docs == n_fit
    assert ctx.cg.addr.n == n_fit
    assert not set(ctx.idx["train_fit"]) & set(ctx.idx["val"]) and not set(ctx.idx["calib"]) & set(ctx.idx["test"])


def test_splits_are_entity_level(ctx):
    """Every candidate pair belongs to exactly one split via its Source-1 entity: no pair-level leakage."""
    e = ctx.cand["e"].values
    owners = sum(np.isin(e, ctx.idx[k]).astype(int) for k in ("train_fit", "calib", "val", "test"))
    assert (owners == 1).all()


def test_only_competition_fields_used(ctx):
    assert list(ctx.ents.columns) == ["entity_id", "name", "address", "city", "region", "postal", "country", "full_address"]
    for c in ctx.F.columns:
        assert not any(t in c.lower() for t in FORBIDDEN_FEATURE_TOKENS), c
    assert set(ctx.ents["country"].dropna()) <= {"US", "India"}          # synthetic train countries


def test_candidates_and_features_sane(ctx):
    assert len(ctx.cand) > 0 and not ctx.cand.duplicated(["e", "r"]).any()
    assert ctx.F.shape[0] == len(ctx.cand)
    assert np.isfinite(ctx.F.select_dtypes("number").fillna(0).values).all()
    assert ctx.y.sum() / len(ctx.true_pairs) > 0.9


def test_model_and_reference_decision(ctx):
    m, s, cols = train_model(ctx, load_cfg("model"))
    assert s.shape == (len(ctx.cand),) and 0 <= s.min() and s.max() <= 1
    ev = reference_eval(ctx, s)
    for mech in ("top1_t", "multi"):
        assert 0.0 <= ev[mech]["strata"]["overall"]["macro_f05"] <= 1.0


def test_validation_fails_loudly_on_duplicate_ids(ctx):
    ents = ctx.ents.copy()
    ents.loc[1, "entity_id"] = ents.loc[0, "entity_id"]
    with pytest.raises(SchemaError):
        validate(ents, ctx.recs, ctx.labels, {}, "x")
