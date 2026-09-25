"""Entity-level splits. Every learned statistic is fit on `train_fit` only (see CONFIG.md).

train_fit : fits TF-IDF/IDF tables, FS m/u, GBDT, hard-negative mining
calib     : carved from TRAIN; isotonic calibration + decision-parameter tuning
val       : decision-mechanism / component selection (A/B decisions)
test      : touched once per reported system, for the final holdout number
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_splits(entity_ids, seed: int, val_frac: float, test_frac: float, calib_frac_of_train: float):
    ids = np.array(sorted(entity_ids))
    r = np.random.default_rng(seed)
    r.shuffle(ids)
    n = len(ids)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    test, val, train = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]
    n_cal = int(len(train) * calib_frac_of_train)
    calib, train_fit = train[:n_cal], train[n_cal:]
    s = pd.Series("train_fit", index=ids)
    s[calib] = "calib"
    s[val] = "val"
    s[test] = "test"
    assert not (set(train_fit) & set(calib)) and not (set(train_fit) & set(val)) and not (set(val) & set(test))
    return s.rename("split")
