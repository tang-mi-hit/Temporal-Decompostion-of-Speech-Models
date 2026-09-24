import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from speech_strf.model_summary import (
    save_predictability_summary,
    summarize_predictability,
)


def _results(full_by_layer, acoustic_by_layer):
    rows = []
    for layer, full_values in full_by_layer.items():
        for fold, full_r2 in enumerate(full_values):
            rows.append(
                {
                    "layer": layer,
                    "outer_fold": fold,
                    "feature_family": "full",
                    "full_r2": full_r2,
                    "conditional_delta_r2": np.nan,
                }
            )
            rows.append(
                {
                    "layer": layer,
                    "outer_fold": fold,
                    "feature_family": "acoustic",
                    "full_r2": full_r2,
                    "conditional_delta_r2": acoustic_by_layer[layer][fold],
                }
            )
    return pd.DataFrame(rows)


def test_cross_model_summary_is_hubert_relative_and_fold_preserving(tmp_path):
    results = {
        "hubert_large_reference": _results(
            {"layer_00": [0.2, 0.4], "layer_01": [0.5, 0.7]},
            {"layer_00": [0.10, 0.12], "layer_01": [0.20, 0.22]},
        ),
        "wavlm_base_plus": _results(
            {"layer_00": [0.3, 0.5], "layer_01": [0.4, 0.6]},
            {"layer_00": [0.15, 0.17], "layer_01": [0.18, 0.20]},
        ),
    }
    metadata = {
        "hubert_large_reference": {
            "model_id": "facebook/hubert-large-ls960-ft",
            "input_modality": "audio",
        },
        "wavlm_base_plus": {
            "model_id": "microsoft/wavlm-base-plus",
            "input_modality": "audio",
        },
    }
    model, layer, family, folds = summarize_predictability(results, metadata)
    candidate = model.set_index("model_key").loc["wavlm_base_plus"]
    assert candidate["best_layer"] == "layer_01"
    assert np.isclose(candidate["best_layer_mean_full_r2"], 0.5)
    assert np.isclose(candidate["best_full_r2_difference_vs_hubert"], -0.1)
    assert np.isclose(candidate["mean_full_r2_across_layers"], 0.45)
    assert len(folds[folds.model_key == "wavlm_base_plus"]) == 2
    candidate_family = family[
        (family.model_key == "wavlm_base_plus")
        & (family.feature_family == "acoustic")
    ].iloc[0]
    assert np.isclose(candidate_family["mean_delta_r2_across_layers"], 0.175)
    assert {"relative_depth", "sd_full_r2_across_folds"} <= set(layer)

    source = tmp_path / "source.csv"
    source.write_text("test")
    save_predictability_summary(
        tmp_path / "summary",
        model,
        layer,
        family,
        folds,
        {"test": source},
    )
    expected = {
        "model_comparison_summary.csv",
        "model_layer_summary.csv",
        "model_family_summary.csv",
        "paired_best_layer_fold_differences.csv",
        "model_predictability.svg",
        "model_predictability.pdf",
        "model_family_contributions.svg",
        "model_family_contributions.pdf",
        "summary_metadata.json",
    }
    assert expected <= {path.name for path in (tmp_path / "summary").iterdir()}


def test_summary_cli_rejects_model_without_comparability_report(tmp_path):
    reference_contract = tmp_path / "reference_contract.json"
    reference_contract.write_text(
        json.dumps(
            {"locked_reference": {"model_key": "hubert_large_reference"}}
        )
    )
    reference_results = tmp_path / "reference.csv"
    _results(
        {"layer_00": [0.2, 0.3]}, {"layer_00": [0.1, 0.1]}
    ).to_csv(reference_results, index=False)
    project = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_model_comparison.py",
            "--model",
            "wavlm_base_plus",
            "--model-root",
            f"wavlm_base_plus={tmp_path / 'candidate'}",
            "--reference-contract",
            str(reference_contract),
            "--reference-results",
            str(reference_results),
        ],
        cwd=project,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "has no comparability report" in completed.stderr

