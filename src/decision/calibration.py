"""Isotonic calibration fit on the CALIB fold (carved from train), never on val/test."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def fit(self, s: np.ndarray, y: np.ndarray):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
        return self

    def transform(self, s: np.ndarray) -> np.ndarray:
        # tiny raw-score tie-breaker keeps within-entity ordering after isotonic flattening
        return self.iso.predict(s) + 1e-6 * (s - s.min()) / (np.ptp(s) + 1e-12)
