"""Leakage-safe, recording-level Stage 4 encoding models.

The public entry points accept either in-memory recording dictionaries or the
canonical feature NPZ / activation HDF5 files produced by this package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .design_matrix import lagged_design


FAMILIES = ("acoustic", "prosodic", "phonetic", "word", "onset")
CAPACITY_FAMILIES = FAMILIES[:4]


def load_stage4_recordings(
    feature_paths: Mapping[str, str | Path] | str | Path,
    activation_store: str | Path,
    layer: str,
    *,
    recording_ids: Sequence[str] | None = None,
    durations: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load canonical feature NPZs and one canonical HDF5 target layer."""
    with h5py.File(activation_store, "r") as store:
        ids = (
            [str(value) for value in recording_ids]
            if recording_ids is not None
            else sorted(str(value) for value in store)
        )
        result: dict[str, dict[str, Any]] = {}
        for recording_id in ids:
            path = (
                Path(feature_paths) / f"{recording_id}.npz"
                if not isinstance(feature_paths, Mapping)
                else Path(feature_paths[recording_id])
            )
            with np.load(path, allow_pickle=False) as archive:
                required = {"matrix", "times", "names", "families"}
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(
                        f"{recording_id}: feature archive lacks {sorted(missing)}"
                    )
                matrix = np.asarray(archive["matrix"], dtype=float)
                times = np.asarray(archive["times"], dtype=float)
                names = archive["names"].astype(str).tolist()
                families = archive["families"].astype(str).tolist()
            if recording_id not in store:
                raise ValueError(f"Activation store is missing {recording_id}")
            group = store[recording_id]
            try:
                target = np.asarray(group["canonical"][layer][:], dtype=float)
                target_times = np.asarray(group["canonical_timestamps"][:], dtype=float)
            except KeyError as exc:
                raise ValueError(
                    f"{recording_id}: canonical target layer {layer!r} is missing"
                ) from exc
            if matrix.ndim != 2 or target.ndim != 2:
                raise ValueError(f"{recording_id}: predictors and targets must be 2D")
            if matrix.shape != (len(times), len(names)) or len(names) != len(families):
                raise ValueError(f"{recording_id}: invalid feature archive dimensions")
            if target.shape[0] != len(target_times):
                raise ValueError(f"{recording_id}: invalid canonical target dimensions")
            if times.shape != target_times.shape or not np.allclose(
                times, target_times, rtol=0, atol=1e-8
            ):
                raise ValueError(
                    f"{recording_id}: canonical feature and target grids differ"
                )
            duration = (
                float(durations[recording_id])
                if durations is not None and recording_id in durations
                else _duration_from_times(times)
            )
            result[recording_id] = {
                "matrix": matrix,
                "targets": target,
                "times": times,
                "names": names,
                "families": families,
                "duration_seconds": duration,
            }
    return result


def _duration_from_times(times: np.ndarray) -> float:
    if len(times) == 0:
        return 0.0
    if len(times) == 1:
        return 0.0
    return float(times[-1] + np.median(np.diff(times)))


def _normalise_recordings(
    recordings: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    rate_hz: float,
) -> dict[str, dict[str, Any]]:
    if isinstance(recordings, Mapping):
        items = [(str(key), value) for key, value in recordings.items()]
    else:
        items = []
        for value in recordings:
            if "recording_id" not in value:
                raise ValueError("Sequence inputs require recording_id")
            items.append((str(value["recording_id"]), value))
    if len(items) < 3:
        raise ValueError("Nested recording-level CV requires at least three recordings")
    normalised: dict[str, dict[str, Any]] = {}
    schema: list[str] | None = None
    for recording_id, value in items:
        matrix_value = value.get(
            "matrix", value.get("features", value.get("feature_matrix"))
        )
        target_value = value.get("targets", value.get("target_matrix", value.get("y")))
        family_value = value.get("families", value.get("feature_families"))
        if matrix_value is None or target_value is None or family_value is None:
            raise ValueError(
                f"{recording_id}: matrix, targets, and families are required"
            )
        matrix = np.asarray(matrix_value, dtype=float)
        targets = np.asarray(target_value, dtype=float)
        families = np.asarray(family_value).astype(str).tolist()
        if matrix.ndim != 2 or targets.ndim != 2:
            raise ValueError(f"{recording_id}: matrix and targets must be 2D")
        if len(matrix) != len(targets) or matrix.shape[1] != len(families):
            raise ValueError(f"{recording_id}: predictor/target dimensions disagree")
        if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
            raise ValueError(f"{recording_id}: values must be finite")
        if set(families) != set(FAMILIES):
            raise ValueError(
                f"{recording_id}: expected exactly the five families {list(FAMILIES)}"
            )
        if schema is not None and families != schema:
            raise ValueError(f"{recording_id}: feature family schema differs")
        schema = families
        times = np.asarray(
            value.get("times", np.arange(len(matrix), dtype=float) / rate_hz),
            dtype=float,
        )
        if times.ndim != 1 or len(times) != len(matrix):
            raise ValueError(f"{recording_id}: times must match matrix rows")
        if not np.isfinite(times).all() or (
            len(times) > 1 and np.any(np.diff(times) <= 0)
        ):
            raise ValueError(f"{recording_id}: times must be finite and increasing")
        normalised[recording_id] = {
            "matrix": matrix,
            "targets": targets,
            "families": families,
            "times": times,
            "duration_seconds": float(
                value.get("duration_seconds", _duration_from_times(times))
            ),
        }
    target_widths = {value["targets"].shape[1] for value in normalised.values()}
    if len(target_widths) != 1:
        raise ValueError("Target dimensions differ between recordings")
    return dict(sorted(normalised.items()))


def _outer_splits(
    ids: list[str],
    frame_counts: Mapping[str, int],
    sensitivity_groups: Mapping[str, Any] | None,
    sensitivity_folds: int | None,
) -> list[tuple[str, int, list[str], list[str]]]:
    splits = [
        ("primary", fold, [other for other in ids if other != held], [held])
        for fold, held in enumerate(ids)
    ]
    if sensitivity_folds is None:
        return splits
    if sensitivity_folds < 2:
        raise ValueError("sensitivity_folds must be at least two or None")
    if sensitivity_groups is not None:
        missing = set(ids) - set(sensitivity_groups)
        if missing:
            raise ValueError(f"Sensitivity groups missing recordings {sorted(missing)}")
        labels_by_recording = {
            value: sensitivity_groups[value] for value in ids
        }
    else:
        labels_by_recording = {value: value for value in ids}
    expanded_ids = np.concatenate(
        [np.repeat(value, frame_counts[value]) for value in ids]
    )
    labels = np.concatenate(
        [
            np.repeat(labels_by_recording[value], frame_counts[value])
            for value in ids
        ]
    )
    group_count = len(np.unique(labels))
    if group_count < 2:
        raise ValueError("Sensitivity analysis requires at least two outer groups")
    splitter = GroupKFold(n_splits=min(int(sensitivity_folds), group_count))
    for fold, (train, test) in enumerate(
        splitter.split(np.zeros(len(expanded_ids)), groups=labels)
    ):
        train_set = set(expanded_ids[train])
        test_set = set(expanded_ids[test])
        splits.append(
            (
                "sensitivity",
                fold,
                [value for value in ids if value in train_set],
                [value for value in ids if value in test_set],
            )
        )
    return splits


def _fit_target_projection(
    records: dict[str, dict[str, Any]],
    train_ids: list[str],
    eval_ids: list[str],
    requested: int | None,
    seed: int,
    svd_solver: str = "full",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    train = np.vstack([records[value]["targets"] for value in train_ids])
    if requested is None:
        return (
            {value: records[value]["targets"] for value in train_ids},
            {value: records[value]["targets"] for value in eval_ids},
            {
                "units": "original_target_dimensions",
                "target_units_before_pca": int(train.shape[1]),
                "requested_components": None,
                "achieved_components": int(train.shape[1]),
                "total_explained_variance_ratio": 1.0,
                "cumulative_explained_variance_ratio": [1.0],
                "train_recording_ids": list(train_ids),
                "test_recording_ids": list(eval_ids),
            },
        )
    maximum = min(train.shape[1], max(1, len(train) - 1))
    achieved = min(int(requested), maximum)
    if achieved < 1:
        raise ValueError("target_pca_components must be positive")
    projector = PCA(n_components=achieved, svd_solver=svd_solver, random_state=seed)
    projector.fit(train)
    curve = np.cumsum(projector.explained_variance_ratio_)
    train_values = {
        value: projector.transform(records[value]["targets"])[:, :achieved]
        for value in train_ids
    }
    eval_values = {
        value: projector.transform(records[value]["targets"])[:, :achieved]
        for value in eval_ids
    }
    return train_values, eval_values, {
        "units": "target_pca_score",
        "target_units_before_pca": int(train.shape[1]),
        "requested_components": int(requested),
        "svd_solver": svd_solver,
        "achieved_components": int(achieved),
        "total_explained_variance_ratio": float(curve[achieved - 1]),
        "cumulative_explained_variance_ratio": curve.tolist(),
        "train_recording_ids": list(train_ids),
        "test_recording_ids": list(eval_ids),
    }


def _capacity_transform(
    records: dict[str, dict[str, Any]],
    train_ids: list[str],
    eval_ids: list[str],
    enabled: bool,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, Any]]:
    families = next(iter(records.values()))["families"]
    if not enabled:
        return (
            {value: records[value]["matrix"] for value in train_ids + eval_ids},
            list(families),
            {"enabled": False, "rank": None, "families": {}},
        )
    masks = {
        family: np.asarray([value == family for value in families])
        for family in FAMILIES
    }
    stacked = {
        family: np.vstack(
            [records[value]["matrix"][:, masks[family]] for value in train_ids]
        )
        for family in CAPACITY_FAMILIES
    }
    available = {
        family: int(np.linalg.matrix_rank(values))
        for family, values in stacked.items()
    }
    rank = min(3, *available.values())
    if rank < 1:
        raise ValueError("Capacity matching requires positive training rank in each family")
    # sklearn requires at least two input columns.  A zero column preserves
    # the exact decomposition for genuinely one-dimensional families.
    reducer_inputs = {
        family: (
            values
            if values.shape[1] >= 2
            else np.column_stack([values, np.zeros(len(values))])
        )
        for family, values in stacked.items()
    }
    reducers = {
        family: TruncatedSVD(
            n_components=rank, algorithm="randomized", n_iter=7, random_state=seed
        ).fit(reducer_inputs[family])
        for family in CAPACITY_FAMILIES
    }
    transformed: dict[str, np.ndarray] = {}
    output_families: list[str] = []
    for family in CAPACITY_FAMILIES:
        output_families.extend([family] * rank)
    output_families.extend(["onset"] * int(masks["onset"].sum()))
    for recording_id in train_ids + eval_ids:
        matrix = records[recording_id]["matrix"]
        blocks = []
        for family in CAPACITY_FAMILIES:
            values = matrix[:, masks[family]]
            if values.shape[1] == 1:
                values = np.column_stack([values, np.zeros(len(values))])
            blocks.append(reducers[family].transform(values))
        blocks.append(matrix[:, masks["onset"]])
        transformed[recording_id] = np.column_stack(blocks)
    report = {
        "enabled": True,
        "rank": int(rank),
        "train_recording_ids": list(train_ids),
        "test_recording_ids": list(eval_ids),
        "families": {
            family: {
                "input_columns": int(masks[family].sum()),
                "available_training_rank": available[family],
                "rank": int(rank),
                "explained_variance_coverage": float(
                    reducers[family].explained_variance_ratio_.sum()
                ),
            }
            for family in CAPACITY_FAMILIES
        },
        "onset": {
            "input_columns": int(masks["onset"].sum()),
            "output_columns": int(masks["onset"].sum()),
            "reduced": False,
        },
    }
    return transformed, output_families, report


def _designs(
    matrices: Mapping[str, np.ndarray],
    families: list[str],
    ids: list[str],
    lags_seconds: Sequence[float],
    rate_hz: float,
    pre_shifted: bool,
) -> tuple[dict[str, np.ndarray], list[str]]:
    if pre_shifted:
        return {value: matrices[value] for value in ids}, list(families)
    output: dict[str, np.ndarray] = {}
    lagged_families: list[str] | None = None
    for recording_id in ids:
        matrix = matrices[recording_id]
        design, current = lagged_design(
            matrix,
            np.full(len(matrix), recording_id, dtype=object),
            families,
            list(lags_seconds),
            rate_hz,
        )
        output[recording_id] = design
        if lagged_families is not None and current != lagged_families:
            raise RuntimeError("Lagged feature schema changed between recordings")
        lagged_families = current
    return output, lagged_families or []


def _fit_predictor_preparation(
    records: dict[str, dict[str, Any]],
    train_ids: list[str],
    eval_ids: list[str],
    *,
    capacity_mode: bool,
    seed: int,
    lags_seconds: Sequence[float],
    rate_hz: float,
    pre_shifted: bool,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, Any]]:
    matrices, families, report = _capacity_transform(
        records, train_ids, eval_ids, capacity_mode, seed
    )
    designs, lagged_families = _designs(
        matrices,
        families,
        train_ids + eval_ids,
        lags_seconds,
        rate_hz,
        pre_shifted,
    )
    return designs, lagged_families, report


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred, multioutput="variance_weighted"))


def _choose_alpha(
    records: dict[str, dict[str, Any]],
    outer_train_ids: list[str],
    family: str | None,
    alphas: Sequence[float],
    *,
    target_pca_components: int | None,
    capacity_mode: bool,
    seed: int,
    lags_seconds: Sequence[float],
    rate_hz: float,
    pre_shifted: bool,
    inner_folds: int,
) -> tuple[float, dict[float, float]]:
    values = {float(value): [] for value in alphas}
    if not values or any(value < 0 or not np.isfinite(value) for value in values):
        raise ValueError("alphas must be a non-empty sequence of finite nonnegative values")
    expanded_ids = np.concatenate(
        [
            np.repeat(recording_id, len(records[recording_id]["matrix"]))
            for recording_id in outer_train_ids
        ]
    )
    splitter = GroupKFold(
        n_splits=min(int(inner_folds), len(outer_train_ids))
    )
    for inner_train_rows, inner_test_rows in splitter.split(
        np.zeros(len(expanded_ids)), groups=expanded_ids
    ):
        train_set = set(expanded_ids[inner_train_rows])
        test_set = set(expanded_ids[inner_test_rows])
        inner_train = [
            value for value in outer_train_ids if value in train_set
        ]
        inner_test_ids = [
            value for value in outer_train_ids if value in test_set
        ]
        designs, families, _ = _fit_predictor_preparation(
            records,
            inner_train,
            inner_test_ids,
            capacity_mode=capacity_mode,
            seed=seed,
            lags_seconds=lags_seconds,
            rate_hz=rate_hz,
            pre_shifted=pre_shifted,
        )
        y_train, y_test, _ = _fit_target_projection(
            records, inner_train, inner_test_ids, target_pca_components, seed
        )
        keep = np.ones(len(families), dtype=bool)
        if family is not None:
            keep = np.asarray([value != family for value in families])
        x_train = np.vstack([designs[value][:, keep] for value in inner_train])
        target_train = np.vstack([y_train[value] for value in inner_train])
        for alpha in values:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train, target_train)
            values[alpha].extend(
                _score(
                    y_test[inner_test],
                    model.predict(designs[inner_test][:, keep]),
                )
                for inner_test in inner_test_ids
            )
    means = {alpha: float(np.mean(scores)) for alpha, scores in values.items()}
    # Explicit tie-breaking makes alpha choice deterministic.
    return max(sorted(means), key=lambda alpha: means[alpha]), means


def fit_stage4_encoding(
    recordings: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    alphas: Sequence[float] = (0.1, 1.0, 10.0),
    lags_seconds: Sequence[float] = (0.0,),
    rate_hz: float = 50.0,
    target_pca_components: int | None = None,
    capacity_mode: bool = False,
    reduced_families: Sequence[str] = FAMILIES,
    pre_shifted: bool = False,
    sensitivity_groups: Mapping[str, Any] | None = None,
    sensitivity_folds: int | None = 5,
    inner_folds: int = 4,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Fit independently nested full/reduced models and score each recording.

    Primary folds are leave-one-recording-out.  If ``sensitivity_groups`` is
    supplied, its labels remain together in grouped sensitivity folds;
    otherwise recording IDs are the groups. Set ``sensitivity_folds=None`` to
    omit sensitivity results. Every
    transform (capacity SVD, target PCA, predictor scaling) is fitted only on
    the training recordings of its current inner or outer split.
    """
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least two")
    selected_families = tuple(str(value) for value in reduced_families)
    if (
        not selected_families
        or len(selected_families) != len(set(selected_families))
        or not set(selected_families).issubset(FAMILIES)
    ):
        raise ValueError(
            f"reduced_families must be unique members of {list(FAMILIES)}"
        )
    data = _normalise_recordings(recordings, rate_hz)
    ids = list(data)
    if not pre_shifted and not lags_seconds:
        raise ValueError("lags_seconds must be non-empty")
    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    sensitivity_predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    split_reports: list[dict[str, Any]] = []
    pca_reports: list[dict[str, Any]] = []
    capacity_reports: list[dict[str, Any]] = []

    for split_kind, fold, train_ids, test_ids in _outer_splits(
        ids,
        {value: len(data[value]["matrix"]) for value in ids},
        sensitivity_groups,
        sensitivity_folds,
    ):
        if len(train_ids) < 2:
            raise ValueError(
                f"{split_kind} fold {fold} leaves fewer than two inner-CV recordings"
            )
        designs, lagged_families, capacity_report = _fit_predictor_preparation(
            data,
            train_ids,
            test_ids,
            capacity_mode=capacity_mode,
            seed=random_seed,
            lags_seconds=lags_seconds,
            rate_hz=rate_hz,
            pre_shifted=pre_shifted,
        )
        y_train, y_test, pca_report = _fit_target_projection(
            data, train_ids, test_ids, target_pca_components, random_seed
        )
        pca_report.update({"split_kind": split_kind, "outer_fold": fold})
        capacity_report.update({"split_kind": split_kind, "outer_fold": fold})
        pca_reports.append(pca_report)
        capacity_reports.append(capacity_report)
        split_reports.append(
            {
                "split_kind": split_kind,
                "outer_fold": fold,
                "train_recording_ids": list(train_ids),
                "test_recording_ids": list(test_ids),
            }
        )
        x_train_full = np.vstack([designs[value] for value in train_ids])
        target_train = np.vstack([y_train[value] for value in train_ids])

        full_alpha, full_cv = _choose_alpha(
            data,
            train_ids,
            None,
            alphas,
            target_pca_components=target_pca_components,
            capacity_mode=capacity_mode,
            seed=random_seed,
            lags_seconds=lags_seconds,
            rate_hz=rate_hz,
            pre_shifted=pre_shifted,
            inner_folds=inner_folds,
        )
        full = make_pipeline(StandardScaler(), Ridge(alpha=full_alpha))
        full.fit(x_train_full, target_train)
        full_predictions = {
            recording_id: full.predict(designs[recording_id])
            for recording_id in test_ids
        }
        reduced_alphas: dict[str, float] = {}
        for family in selected_families:
            reduced_alpha, reduced_cv = _choose_alpha(
                data,
                train_ids,
                family,
                alphas,
                target_pca_components=target_pca_components,
                capacity_mode=capacity_mode,
                seed=random_seed,
                lags_seconds=lags_seconds,
                rate_hz=rate_hz,
                pre_shifted=pre_shifted,
                inner_folds=inner_folds,
            )
            reduced_alphas[family] = reduced_alpha
            keep = np.asarray([value != family for value in lagged_families])
            reduced = make_pipeline(StandardScaler(), Ridge(alpha=reduced_alpha))
            reduced.fit(x_train_full[:, keep], target_train)
            destination = (
                predictions if split_kind == "primary" else sensitivity_predictions
            )
            for recording_id in test_ids:
                full_prediction = full_predictions[recording_id]
                reduced_prediction = reduced.predict(designs[recording_id][:, keep])
                full_r2 = _score(y_test[recording_id], full_prediction)
                reduced_r2 = _score(y_test[recording_id], reduced_prediction)
                destination.setdefault(recording_id, {})[family] = {
                    "target": y_test[recording_id].copy(),
                    "full": full_prediction,
                    "reduced": reduced_prediction,
                }
                rows.append(
                    {
                        "split_kind": split_kind,
                        "outer_fold": fold,
                        "recording_id": recording_id,
                        "family": family,
                        "full_alpha": full_alpha,
                        "reduced_alpha": reduced_alpha,
                        "full_r2": full_r2,
                        "reduced_r2": reduced_r2,
                        "delta_r2": full_r2 - reduced_r2,
                        "frames": int(len(y_test[recording_id])),
                        "duration_seconds": data[recording_id]["duration_seconds"],
                        "train_recording_ids": tuple(train_ids),
                        "test_recording_ids": tuple(test_ids),
                        "full_inner_cv_scores": full_cv,
                        "reduced_inner_cv_scores": reduced_cv,
                    }
                )
        split_reports[-1]["full_alpha"] = full_alpha
        split_reports[-1]["full_inner_cv_scores"] = full_cv
        split_reports[-1]["reduced_alphas"] = reduced_alphas
    return {
        "scores": pd.DataFrame(rows),
        "predictions": predictions,
        "sensitivity_predictions": sensitivity_predictions,
        "split_reports": split_reports,
        "pca_reports": pca_reports,
        "capacity_reports": capacity_reports,
    }


run_stage4_encoding = fit_stage4_encoding


def fit_stage4_fixed_alpha_grouped(
    recordings: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    fixed_alphas: Mapping[int, Mapping[str, float]],
    lags_seconds: Sequence[float],
    rate_hz: float = 50.0,
    target_pca_components: int | None = 30,
    capacity_mode: bool = False,
    reduced_families: Sequence[str] = FAMILIES,
    sensitivity_groups: Mapping[str, Any] | None = None,
    outer_folds: int = 5,
    random_seed: int = 0,
    alpha_source: str | None = None,
    target_pca_solver: str = "auto",
) -> dict[str, Any]:
    """Fit grouped outer folds with prespecified alphas and no inner CV.

    Scores remain recording-specific even when an outer fold contains several
    recordings. ``fixed_alphas`` maps each outer-fold integer to ``full`` and
    one value per requested reduced family.
    """
    if outer_folds < 2:
        raise ValueError("outer_folds must be at least two")
    selected_families = tuple(str(value) for value in reduced_families)
    if (
        not selected_families
        or len(selected_families) != len(set(selected_families))
        or not set(selected_families).issubset(FAMILIES)
    ):
        raise ValueError(
            f"reduced_families must be unique members of {list(FAMILIES)}"
        )
    data = _normalise_recordings(recordings, rate_hz)
    ids = list(data)
    grouped_splits = [
        split
        for split in _outer_splits(
            ids,
            {value: len(data[value]["matrix"]) for value in ids},
            sensitivity_groups,
            outer_folds,
        )
        if split[0] == "sensitivity"
    ]
    expected_fold_ids = {fold for _, fold, _, _ in grouped_splits}
    observed_fold_ids = {int(value) for value in fixed_alphas}
    if observed_fold_ids != expected_fold_ids:
        raise ValueError(
            "Fixed-alpha folds differ from grouped outer folds: "
            f"expected={sorted(expected_fold_ids)}, "
            f"observed={sorted(observed_fold_ids)}"
        )

    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    split_reports: list[dict[str, Any]] = []
    pca_reports: list[dict[str, Any]] = []
    capacity_reports: list[dict[str, Any]] = []
    for _, fold, train_ids, test_ids in grouped_splits:
        alpha_map = {str(key): float(value) for key, value in fixed_alphas[fold].items()}
        required = {"full", *selected_families}
        if set(alpha_map) != required or any(
            not np.isfinite(value) or value < 0 for value in alpha_map.values()
        ):
            raise ValueError(
                f"Fold {fold} fixed alphas must contain exactly {sorted(required)}"
            )
        designs, lagged_families, capacity_report = _fit_predictor_preparation(
            data,
            train_ids,
            test_ids,
            capacity_mode=capacity_mode,
            seed=random_seed,
            lags_seconds=lags_seconds,
            rate_hz=rate_hz,
            pre_shifted=False,
        )
        y_train, y_test, pca_report = _fit_target_projection(
            data,
            train_ids,
            test_ids,
            target_pca_components,
            random_seed,
            svd_solver=target_pca_solver,
        )
        pca_report.update(
            {
                "split_kind": "primary_fast_grouped",
                "outer_fold": fold,
            }
        )
        capacity_report.update(
            {
                "split_kind": "primary_fast_grouped",
                "outer_fold": fold,
            }
        )
        pca_reports.append(pca_report)
        capacity_reports.append(capacity_report)
        x_train = np.vstack([designs[value] for value in train_ids])
        target_train = np.vstack([y_train[value] for value in train_ids])
        full = make_pipeline(StandardScaler(), Ridge(alpha=alpha_map["full"]))
        full.fit(x_train, target_train)
        full_predictions = {
            recording_id: full.predict(designs[recording_id])
            for recording_id in test_ids
        }
        for family in selected_families:
            keep = np.asarray([value != family for value in lagged_families])
            reduced = make_pipeline(
                StandardScaler(), Ridge(alpha=alpha_map[family])
            )
            reduced.fit(x_train[:, keep], target_train)
            for recording_id in test_ids:
                full_prediction = full_predictions[recording_id]
                reduced_prediction = reduced.predict(
                    designs[recording_id][:, keep]
                )
                full_r2 = _score(y_test[recording_id], full_prediction)
                reduced_r2 = _score(y_test[recording_id], reduced_prediction)
                predictions.setdefault(recording_id, {})[family] = {
                    "target": y_test[recording_id].copy(),
                    "full": full_prediction,
                    "reduced": reduced_prediction,
                }
                rows.append(
                    {
                        "split_kind": "primary_fast_grouped",
                        "outer_fold": fold,
                        "recording_id": recording_id,
                        "family": family,
                        "full_alpha": alpha_map["full"],
                        "reduced_alpha": alpha_map[family],
                        "alpha_selection": "fixed_from_existing_original",
                        "full_r2": full_r2,
                        "reduced_r2": reduced_r2,
                        "delta_r2": full_r2 - reduced_r2,
                        "frames": int(len(y_test[recording_id])),
                        "duration_seconds": data[recording_id]["duration_seconds"],
                        "train_recording_ids": tuple(train_ids),
                        "test_recording_ids": tuple(test_ids),
                    }
                )
        split_reports.append(
            {
                "split_kind": "primary_fast_grouped",
                "outer_fold": fold,
                "train_recording_ids": list(train_ids),
                "test_recording_ids": list(test_ids),
                "full_alpha": alpha_map["full"],
                "reduced_alphas": {
                    family: alpha_map[family] for family in selected_families
                },
                "alpha_selection": "fixed_from_existing_original",
                "alpha_source": alpha_source,
                "inner_cv_repeated": False,
            }
        )
    return {
        "scores": pd.DataFrame(rows),
        "predictions": predictions,
        "sensitivity_predictions": {},
        "split_reports": split_reports,
        "pca_reports": pca_reports,
        "capacity_reports": capacity_reports,
    }

