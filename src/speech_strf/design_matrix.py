from __future__ import annotations

import numpy as np


def lagged_design(
    matrix: np.ndarray,
    groups: np.ndarray,
    feature_families: list[str],
    lags_seconds: list[float],
    rate_hz: float,
) -> tuple[np.ndarray, list[str]]:
    matrix, groups = np.asarray(matrix), np.asarray(groups)
    if matrix.shape[0] != len(groups) or matrix.shape[1] != len(feature_families):
        raise ValueError("Design, groups, and feature family dimensions do not agree")
    blocks, lagged_families = [], []
    for lag in lags_seconds:
        shift = int(round(lag * rate_hz))
        block = np.zeros_like(matrix)
        destination = np.arange(len(matrix))
        source = destination - shift
        valid = (source >= 0) & (source < len(matrix))
        valid &= np.where(valid, groups[np.clip(source, 0, len(matrix) - 1)] == groups, False)
        block[valid] = matrix[source[valid]]
        blocks.append(block)
        lagged_families.extend(feature_families)
    return np.column_stack(blocks), lagged_families


def reduced_columns(families: list[str], omitted_family: str) -> np.ndarray:
    return np.asarray([family != omitted_family for family in families], dtype=bool)

