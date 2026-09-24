"""Recording-level inference for the Stage 4 STRF analysis.

Layers and checkpoints are measurements within a recording, not independent
replicates.  :func:`analyze_stage4_statistics` therefore averages them before
performing any inference across recordings.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd


MIN_BOOTSTRAP_SAMPLES = 10_000
EXPECTED_FAMILY_COUNT = 5


def _one_dimensional_finite(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    bad = np.flatnonzero(~np.isfinite(array))
    if bad.size:
        raise ValueError(
            f"{name} contains non-finite values at positions {bad.tolist()}"
        )
    return array


def paired_differences(
    observed: Sequence[float] | np.ndarray,
    mean_null: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return the recording-wise estimand ``observed - mean(null)``.

    ``mean_null`` may have the same one-dimensional shape as ``observed`` or
    may be an ``(recording, null_draw)`` matrix.  In the latter case its
    arithmetic mean is taken within each recording.
    """
    observed_array = _one_dimensional_finite(observed, "observed")
    if mean_null is None:
        return observed_array.copy()
    null_array = np.asarray(mean_null, dtype=float)
    if null_array.ndim == 2:
        if null_array.shape[0] != observed_array.size or null_array.shape[1] == 0:
            raise ValueError(
                "mean_null matrix must have one non-empty row per recording"
            )
        if not np.isfinite(null_array).all():
            locations = np.argwhere(~np.isfinite(null_array)).tolist()
            raise ValueError(f"mean_null contains non-finite values at {locations}")
        null_array = null_array.mean(axis=1)
    else:
        null_array = _one_dimensional_finite(null_array, "mean_null")
    if null_array.shape != observed_array.shape:
        raise ValueError("observed and mean_null must have matching recordings")
    return observed_array - null_array


def exact_paired_sign_flip(
    observed: Sequence[float] | np.ndarray,
    mean_null: Sequence[float] | np.ndarray | None = None,
    *,
    estimand: str | None = None,
) -> dict[str, int | float | str]:
    """Run an exact paired, two-sided sign-flip test.

    All ``2**N`` sign assignments are enumerated.  The p-value is the fraction
    whose absolute arithmetic mean is at least the absolute observed
    all-positive-sign mean.  Thus the observed assignment is included and no
    Monte Carlo or plus-one correction is used.
    """
    differences = paired_differences(observed, mean_null)
    n_recordings = differences.size
    observed_effect = float(sum(differences.tolist()) / n_recordings)
    threshold = abs(observed_effect)
    extreme = 0
    for assignment in product((-1.0, 1.0), repeat=n_recordings):
        statistic = float(
            sum(
                sign * difference
                for sign, difference in zip(assignment, differences)
            )
            / n_recordings
        )
        if abs(statistic) >= threshold:
            extreme += 1
    assignments = 1 << n_recordings
    return {
        "estimand": estimand or (
            "mean_recording_effect"
            if mean_null is None
            else "mean_observed_minus_mean_null"
        ),
        "effect": observed_effect,
        "p_value": extreme / assignments,
        "extreme_assignments": extreme,
        "total_assignments": assignments,
        "n_recordings": int(n_recordings),
        "positive_count": int(np.count_nonzero(differences > 0)),
        "negative_count": int(np.count_nonzero(differences < 0)),
        "zero_count": int(np.count_nonzero(differences == 0)),
    }


def recording_cluster_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    n_bootstrap: int = MIN_BOOTSTRAP_SAMPLES,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Seeded recording-cluster percentile interval for a recording summary."""
    value_array = _one_dimensional_finite(values, "values")
    if n_bootstrap < MIN_BOOTSTRAP_SAMPLES:
        raise ValueError(f"n_bootstrap must be at least {MIN_BOOTSTRAP_SAMPLES}")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    weight_array: np.ndarray | None = None
    if weights is not None:
        weight_array = _one_dimensional_finite(weights, "weights")
        if weight_array.shape != value_array.shape:
            raise ValueError("weights and values must have matching recordings")
        if np.any(weight_array <= 0):
            raise ValueError("weights must be strictly positive")

    generator = np.random.default_rng(seed)
    n = value_array.size
    estimates = np.empty(n_bootstrap, dtype=float)
    # Chunking bounds temporary memory while retaining deterministic draws.
    for start in range(0, n_bootstrap, 4096):
        stop = min(start + 4096, n_bootstrap)
        indices = generator.integers(0, n, size=(stop - start, n))
        sampled = value_array[indices]
        if weight_array is None:
            estimates[start:stop] = sampled.mean(axis=1)
        else:
            sampled_weights = weight_array[indices]
            denominator = sampled_weights.sum(axis=1)
            estimates[start:stop] = (
                sampled * sampled_weights
            ).sum(axis=1) / denominator
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return float(low), float(high)


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    values = _one_dimensional_finite(p_values, "p_values")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _expected_recording_values(
    frame: pd.DataFrame,
    expected_recordings: Sequence[str],
    *,
    recording_col: str,
    value_col: str,
) -> np.ndarray:
    expected = [str(value) for value in expected_recordings]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected_recordings must be non-empty and unique")
    if frame[recording_col].duplicated().any():
        duplicated = sorted(
            frame.loc[frame[recording_col].duplicated(False), recording_col]
            .astype(str)
            .unique()
        )
        raise ValueError(f"duplicate recordings: {duplicated}")
    indexed = frame.assign(
        **{recording_col: frame[recording_col].astype(str)}
    ).set_index(recording_col)
    missing = sorted(set(expected) - set(indexed.index))
    unexpected = sorted(set(indexed.index) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"incomplete expected recording vector; missing={missing}, "
            f"unexpected={unexpected}"
        )
    values = pd.to_numeric(indexed.loc[expected, value_col], errors="coerce").to_numpy()
    bad = [expected[index] for index in np.flatnonzero(~np.isfinite(values))]
    if bad:
        raise ValueError(f"non-finite {value_col} recording values: {bad}")
    return values.astype(float)


def aggregate_layer_effects(
    results: pd.DataFrame,
    *,
    model_col: str = "model",
    recording_col: str = "recording_id",
    family_col: str = "family",
    effect_col: str = "effect",
    observed_col: str = "observed",
    mean_null_col: str = "mean_null",
    duration_col: str = "duration_seconds",
) -> pd.DataFrame:
    """Average all layer/checkpoint measurements within recording-family cells."""
    required = {model_col, recording_col, family_col}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results missing columns: {sorted(missing)}")
    working = results.copy()
    if effect_col not in working:
        null_columns = {observed_col, mean_null_col}
        if not null_columns <= set(working):
            raise ValueError(
                f"results require {effect_col!r}, or both "
                f"{observed_col!r} and {mean_null_col!r}"
            )
        working[effect_col] = pd.to_numeric(
            working[observed_col], errors="coerce"
        ) - pd.to_numeric(working[mean_null_col], errors="coerce")
    working[effect_col] = pd.to_numeric(working[effect_col], errors="coerce")
    nonfinite = ~np.isfinite(working[effect_col].to_numpy(dtype=float))
    if nonfinite.any():
        bad = working.loc[
            nonfinite, [model_col, recording_col, family_col]
        ].drop_duplicates()
        raise ValueError(
            "non-finite layer/checkpoint effects in cells: "
            f"{[tuple(row) for row in bad.itertuples(index=False, name=None)]}"
        )
    keys = [model_col, recording_col, family_col]
    aggregations: dict[str, tuple[str, str]] = {
        effect_col: (effect_col, "mean"),
        "n_layer_checkpoint_measurements": (effect_col, "size"),
    }
    if duration_col in working:
        duration_counts = working.groupby(keys, dropna=False)[duration_col].nunique(
            dropna=False
        )
        if (duration_counts > 1).any():
            bad = [tuple(value) for value in duration_counts[duration_counts > 1].index]
            raise ValueError(f"inconsistent durations within cells: {bad}")
        aggregations[duration_col] = (duration_col, "first")
    return (
        working.groupby(keys, as_index=False, sort=True, dropna=False)
        .agg(**aggregations)
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )


def analyze_stage4_statistics(
    results: pd.DataFrame,
    expected_recordings: Sequence[str],
    families: Sequence[str],
    *,
    n_bootstrap: int = MIN_BOOTSTRAP_SAMPLES,
    seed: int = 0,
    model_col: str = "model",
    recording_col: str = "recording_id",
    family_col: str = "family",
    effect_col: str = "effect",
    duration_col: str = "duration_seconds",
    estimand: str = (
        "equal-recording mean conditional delta R2 across benchmark recordings"
    ),
) -> pd.DataFrame:
    """Produce one inference row per model and each of exactly five families."""
    family_list = [str(value) for value in families]
    if len(family_list) != EXPECTED_FAMILY_COUNT or len(set(family_list)) != len(
        family_list
    ):
        raise ValueError("families must contain exactly five unique values")
    aggregate = aggregate_layer_effects(
        results,
        model_col=model_col,
        recording_col=recording_col,
        family_col=family_col,
        effect_col=effect_col,
        duration_col=duration_col,
    )
    observed_families = set(aggregate[family_col].astype(str))
    if observed_families != set(family_list):
        raise ValueError(
            "family set mismatch; "
            f"missing={sorted(set(family_list) - observed_families)}, "
            f"unexpected={sorted(observed_families - set(family_list))}"
        )

    rows: list[dict[str, object]] = []
    expected = [str(value) for value in expected_recordings]
    for model, model_frame in aggregate.groupby(model_col, sort=True, dropna=False):
        for family in family_list:
            group = model_frame[
                model_frame[family_col].astype(str).eq(family)
            ]
            try:
                values = _expected_recording_values(
                    group,
                    expected,
                    recording_col=recording_col,
                    value_col=effect_col,
                )
            except ValueError as exc:
                raise ValueError(
                    f"model={model!r}, family={family!r}: {exc}"
                ) from exc
            test = exact_paired_sign_flip(values, estimand=estimand)
            equal_ci = recording_cluster_bootstrap_ci(
                values, n_bootstrap=n_bootstrap, seed=seed
            )
            row: dict[str, object] = {
                model_col: model,
                family_col: family,
                **test,
                "ci_low": equal_ci[0],
                "ci_high": equal_ci[1],
                "equal_recording_effect": test["effect"],
                "equal_recording_ci_low": equal_ci[0],
                "equal_recording_ci_high": equal_ci[1],
                "null_hypothesis": (
                    "recording-level paired effects are sign-symmetric around zero"
                ),
                "inference_scope": (
                    "variability and consistency across the fixed benchmark "
                    "recordings; not a narrative-population claim"
                ),
            }
            if duration_col in group:
                durations = _expected_recording_values(
                    group,
                    expected,
                    recording_col=recording_col,
                    value_col=duration_col,
                )
                if np.any(durations <= 0):
                    raise ValueError(
                        f"model={model!r}, family={family!r}: durations must be "
                        "strictly positive"
                    )
                weighted_effect = float(np.average(values, weights=durations))
                weighted_ci = recording_cluster_bootstrap_ci(
                    values,
                    weights=durations,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                row.update(
                    {
                        "duration_weighted_effect": weighted_effect,
                        "duration_weighted_ci_low": weighted_ci[0],
                        "duration_weighted_ci_high": weighted_ci[1],
                    }
                )
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary["q_value"] = np.nan
    for _, indices in summary.groupby(model_col, sort=False).groups.items():
        if len(indices) != EXPECTED_FAMILY_COUNT:
            raise ValueError("BH correction requires exactly five families per model")
        summary.loc[indices, "q_value"] = benjamini_hochberg(
            summary.loc[indices, "p_value"].to_numpy()
        )
    return summary.reset_index(drop=True)


# Explicit aliases make the intended public operations easy to discover.
exact_sign_flip_test = exact_paired_sign_flip
exact_paired_sign_flip_test = exact_paired_sign_flip
bootstrap_percentile_ci = recording_cluster_bootstrap_ci
cluster_bootstrap_ci = recording_cluster_bootstrap_ci
bh_adjust = benjamini_hochberg
benjamini_hochberg_correction = benjamini_hochberg
summarize_stage4_statistics = analyze_stage4_statistics
run_stage4_statistics = analyze_stage4_statistics
