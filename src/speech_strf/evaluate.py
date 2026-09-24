from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def held_out_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred, multioutput="variance_weighted"))


def conditional_contribution(full_r2: float, reduced_r2: float) -> float:
    return float(full_r2 - reduced_r2)


def stable_proportion(delta_r2: float, full_r2: float, epsilon: float = 1e-4) -> float:
    return float(delta_r2 / full_r2) if full_r2 > epsilon else np.nan

