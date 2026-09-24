from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .fit_encoding import grouped_splits
from .provenance import load_config


COMPARABILITY_FIELDS = [
    "stimulus_ids",
    "canonical_rate_hz",
    "canonical_grid_sha256",
    "feature_columns",
    "feature_families",
    "feature_matrix_sha256",
    "feature_config_sha256",
    "feature_preprocessing",
    "analysis_config_sha256",
    "target_preprocessing",
    "lag_grid_seconds",
    "outer_test_groups",
    "cross_validation",
    "grouping_policy",
    "reduced_model_specifications",
    "metrics",
    "statistical_procedures",
    "result_schema",
]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_comparability_contract(
    manifest_path: str | Path,
    features_dir: str | Path,
    feature_config_path: str | Path,
    analysis_config_path: str | Path,
) -> dict:
    manifest = pd.read_csv(manifest_path)
    stimulus_ids = sorted(manifest["recording_id"].astype(str))
    names = families = None
    groups: list[str] = []
    grid_hashes = {}
    matrix_hashes = {}
    for recording_id in stimulus_ids:
        feature = np.load(Path(features_dir) / f"{recording_id}.npz")
        current_names = feature["names"].tolist()
        current_families = feature["families"].tolist()
        if names is None:
            names, families = current_names, current_families
        if names != current_names or families != current_families:
            raise ValueError(f"Feature schema mismatch for {recording_id}")
        grid_hashes[recording_id] = hashlib.sha256(
            np.asarray(feature["times"], dtype=np.float64).tobytes()
        ).hexdigest()
        matrix_hashes[recording_id] = hashlib.sha256(
            np.ascontiguousarray(feature["matrix"]).tobytes()
        ).hexdigest()
        groups.extend([recording_id] * len(feature["times"]))
    analysis_config = load_config(analysis_config_path)
    feature_config = load_config(feature_config_path)
    group_array = np.asarray(groups)
    outer_test_groups = []
    for _, test in grouped_splits(
        group_array, int(analysis_config["analysis"]["outer_folds"])
    ):
        outer_test_groups.append(sorted(set(group_array[test])))
    return {
        "stimulus_ids": stimulus_ids,
        "canonical_rate_hz": analysis_config["analysis_rate_hz"],
        "canonical_grid_sha256": grid_hashes,
        "feature_columns": names,
        "feature_families": families,
        "feature_matrix_sha256": matrix_hashes,
        "feature_config_sha256": file_sha256(feature_config_path),
        "feature_preprocessing": {
            "definition": feature_config,
            "regression_scaling": "StandardScaler fitted on each training fold only",
        },
        "analysis_config_sha256": file_sha256(analysis_config_path),
        "target_preprocessing": {
            "mode": analysis_config["analysis"]["target_mode"],
            "n_components": analysis_config["analysis"]["n_components"],
            "pca_scope": "fit on each training fold only",
        },
        "lag_grid_seconds": analysis_config["analysis"]["lags_seconds"],
        "outer_test_groups": outer_test_groups,
        "cross_validation": {
            "outer_folds": analysis_config["analysis"]["outer_folds"],
            "inner_folds": analysis_config["analysis"]["inner_folds"],
            "alphas": analysis_config["analysis"]["alphas"],
            "splitter": "GroupKFold",
        },
        "grouping_policy": "GroupKFold; recording_id/story_id frames remain together",
        "reduced_model_specifications": [
            {"omitted_family": family, "comparison": "full_minus_reduced"}
            for family in sorted(set(families))
        ],
        "metrics": {
            "full": "held_out_variance_weighted_r2",
            "conditional_unique_contribution": "full_r2_minus_reduced_r2",
            "proportion_guard": "reported_only_when_full_r2_exceeds_configured_epsilon",
        },
        "statistical_procedures": {
            "permutation": None,
            "bootstrap": None,
            "reliability": "outer_fold_distribution",
        },
        "result_schema": [
            "layer",
            "outer_fold",
            "feature_family",
            "alpha",
            "full_r2",
            "reduced_r2",
            "conditional_delta_r2",
            "conditional_proportion",
        ],
    }


def compare_contracts(reference: dict, candidate: dict) -> dict:
    checks = {
        field: {
            "match": reference.get(field) == candidate.get(field),
            "reference": reference.get(field),
            "candidate": candidate.get(field),
        }
        for field in COMPARABILITY_FIELDS
    }
    return {
        "comparable": all(item["match"] for item in checks.values()),
        "checks": checks,
    }


def write_json(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def compare_hubert_regression(
    reference_store: str | Path,
    candidate_store: str | Path,
    reference_results: str | Path,
    candidate_results: str | Path,
    *,
    activation_atol: float = 2e-3,
    timestamp_atol: float = 1e-6,
    summary_atol: float = 1e-4,
) -> dict:
    checks: list[dict] = []
    with h5py.File(reference_store) as reference, h5py.File(candidate_store) as candidate:
        shared_recordings = sorted(set(reference) & set(candidate))
        shared_recordings = [
            name for name in shared_recordings if not name.startswith("__")
        ]
        if not shared_recordings:
            raise ValueError("No shared HuBERT smoke-test recordings")
        for recording_id in shared_recordings:
            ref_group, candidate_group = reference[recording_id], candidate[recording_id]
            ref_times = _timestamps(ref_group)
            candidate_times = _timestamps(candidate_group)
            time_match = ref_times.shape == candidate_times.shape and np.allclose(
                ref_times, candidate_times, atol=timestamp_atol, rtol=0
            )
            checks.append(
                {
                    "recording_id": recording_id,
                    "kind": "timestamps",
                    "match": bool(time_match),
                    "reference_shape": list(ref_times.shape),
                    "candidate_shape": list(candidate_times.shape),
                }
            )
            ref_layers = _layer_names(ref_group)
            candidate_layers = _layer_names(candidate_group)
            checks.append(
                {
                    "recording_id": recording_id,
                    "kind": "layer_names",
                    "match": ref_layers == candidate_layers,
                    "reference": ref_layers,
                    "candidate": candidate_layers,
                }
            )
            for layer in sorted(set(ref_layers) & set(candidate_layers)):
                ref_values = _layer_values(ref_group, layer)
                candidate_values = _layer_values(candidate_group, layer)
                value_match = (
                    ref_values.shape == candidate_values.shape
                    and np.allclose(
                        ref_values, candidate_values, atol=activation_atol, rtol=1e-4
                    )
                )
                checks.append(
                    {
                        "recording_id": recording_id,
                        "kind": "activation",
                        "layer": layer,
                        "match": bool(value_match),
                        "reference_shape": list(ref_values.shape),
                        "candidate_shape": list(candidate_values.shape),
                    }
                )
    reference_rows = _result_rows(reference_results)
    candidate_rows = _result_rows(candidate_results)
    key_columns = ["layer", "outer_fold", "feature_family"]
    aligned = reference_rows.merge(
        candidate_rows,
        on=key_columns,
        suffixes=("_reference", "_candidate"),
        how="outer",
        indicator=True,
    )
    same_keys = bool((aligned["_merge"] == "both").all())
    value_columns = [
        "alpha",
        "full_r2",
        "reduced_r2",
        "conditional_delta_r2",
        "conditional_proportion",
    ]
    summary_match = same_keys and all(
        np.allclose(
            aligned[f"{column}_reference"],
            aligned[f"{column}_candidate"],
            atol=summary_atol,
            rtol=1e-4,
            equal_nan=True,
        )
        for column in value_columns
    )
    checks.append(
        {
            "kind": "downstream_fold_rows",
            "match": bool(summary_match),
            "absolute_tolerance": summary_atol,
            "compared_columns": key_columns + value_columns,
        }
    )
    return {
        "passed": all(check["match"] for check in checks),
        "tolerances": {
            "activation_atol": activation_atol,
            "timestamp_atol": timestamp_atol,
            "summary_atol": summary_atol,
        },
        "checks": checks,
    }


def _timestamps(group) -> np.ndarray:
    if "native_timestamps" in group:
        return group["native_timestamps"][:]
    if "_frame_times_seconds" in group:
        return group["_frame_times_seconds"][:]
    raise KeyError("Activation group has no recognized timestamp dataset")


def _layer_names(group) -> list[str]:
    if "layer_names_json" in group.attrs:
        return json.loads(group.attrs["layer_names_json"])
    container = group["native"] if "native" in group else group
    return sorted(name for name in container if not name.startswith("_"))


def _layer_values(group, layer: str) -> np.ndarray:
    return group["native"][layer][:] if "native" in group else group[layer][:]


def _result_rows(path: str | Path) -> pd.DataFrame:
    columns = [
        "layer",
        "outer_fold",
        "feature_family",
        "alpha",
        "full_r2",
        "reduced_r2",
        "conditional_delta_r2",
        "conditional_proportion",
    ]
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Regression result table lacks columns: {sorted(missing)}")
    return frame[columns].sort_values(
        ["layer", "outer_fold", "feature_family"]
    )
