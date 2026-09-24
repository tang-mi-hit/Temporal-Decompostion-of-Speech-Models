import json

import h5py
import numpy as np
import pandas as pd

from speech_strf.adapters import ActivationResult, StimulusRecord, save_activation_artifacts
from speech_strf.comparability import compare_contracts, compare_hubert_regression


def _contract():
    return {
        "stimulus_ids": ["a", "b"],
        "canonical_rate_hz": 50,
        "feature_columns": ["envelope", "word_onset"],
        "feature_families": ["acoustic", "word"],
        "feature_config_sha256": "feature-hash",
        "feature_preprocessing": {"normalization": "training-fold-only"},
        "lag_grid_seconds": [-0.1, 0, 0.1],
        "outer_test_groups": [["a"], ["b"]],
        "reduced_model_specifications": [
            {"omitted_family": "acoustic"},
            {"omitted_family": "word"},
        ],
        "metrics": {"full": "held_out_variance_weighted_r2"},
        "result_schema": ["model_key", "layer", "outer_fold"],
    }


def test_comparability_detects_split_mismatch():
    reference, candidate = _contract(), _contract()
    assert compare_contracts(reference, candidate)["comparable"]
    candidate["outer_test_groups"] = [["a", "b"]]
    report = compare_contracts(reference, candidate)
    assert not report["comparable"]
    assert not report["checks"]["outer_test_groups"]["match"]


def test_saved_hubert_regression_contract_accepts_tolerance(tmp_path):
    reference_store = tmp_path / "reference.h5"
    candidate_store = tmp_path / "candidate.h5"
    times = np.array([0.01, 0.03, 0.05])
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    with h5py.File(reference_store, "w") as store:
        group = store.create_group("smoke")
        group.attrs["layer_names_json"] = json.dumps(["layer_00_input"])
        group.create_dataset("_frame_times_seconds", data=times)
        group.create_dataset("layer_00_input", data=values)
    result = ActivationResult(
        native_states={"layer_00_input": values + 1e-4},
        native_times=times,
        canonical_states={"layer_00_input": values},
        canonical_times=times,
        metadata={"layers": [{"name": "layer_00_input", "hidden_size": 2}]},
    )
    save_activation_artifacts(
        candidate_store, StimulusRecord("smoke", 0.06), result
    )
    rows = pd.DataFrame(
        [
            {
                "layer": "layer_00_input",
                "outer_fold": 0,
                "feature_family": "full",
                "alpha": 1.0,
                "full_r2": 0.5,
                "reduced_r2": np.nan,
                "conditional_delta_r2": np.nan,
                "conditional_proportion": np.nan,
            },
            {
                "layer": "layer_00_input",
                "outer_fold": 0,
                "feature_family": "acoustic",
                "alpha": 1.0,
                "full_r2": 0.5,
                "reduced_r2": 0.4,
                "conditional_delta_r2": 0.1,
                "conditional_proportion": 0.2,
            },
        ]
    )
    reference_results = tmp_path / "reference.csv"
    candidate_results = tmp_path / "candidate.csv"
    rows.to_csv(reference_results, index=False)
    rows.assign(conditional_delta_r2=lambda x: x.conditional_delta_r2 + 1e-6).to_csv(
        candidate_results, index=False
    )
    report = compare_hubert_regression(
        reference_store,
        candidate_store,
        reference_results,
        candidate_results,
    )
    assert report["passed"]
    assert report["tolerances"]["activation_atol"] == 2e-3

