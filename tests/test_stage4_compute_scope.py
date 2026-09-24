from __future__ import annotations

import json

import pandas as pd
import pytest

from speech_strf.stage4_compute_scope import (
    ALL_SCOPE_MODELS,
    FAMILIES,
    load_legacy_fixed_alphas,
    publish_compute_scope_manifests,
    resolve_depth_layers,
    validate_selected_depth_controls,
)


def test_middle_depth_resolution_is_deterministic_and_lower_tie_wins():
    layers = ["layer_00_input", *[f"layer_{index:02d}" for index in range(1, 13)]]

    assert resolve_depth_layers(layers) == {
        "input": "layer_00_input",
        "middle": "layer_06",
        "final": "layer_12",
    }
    assert resolve_depth_layers(["input", "encoder_1", "encoder_2", "encoder_3"]) == {
        "input": "input",
        "middle": "encoder_1",
        "final": "encoder_3",
    }


def test_legacy_alpha_loader_requires_one_complete_row_per_model_and_fold(tmp_path):
    path = tmp_path / "results.csv"
    rows = []
    for fold in range(5):
        for family in ("full", *FAMILIES):
            rows.append(
                {
                    "layer": "layer_00_input",
                    "outer_fold": fold,
                    "feature_family": family,
                    "alpha": fold + 0.1,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)

    result = load_legacy_fixed_alphas(path, "layer_00_input")

    assert set(result) == set(range(5))
    assert result[3]["full"] == 3.1
    assert set(result[0]) == {"full", *FAMILIES}


def test_compute_scope_manifests_have_exact_deadline_task_counts(tmp_path):
    layers = ["input", *[f"encoder_{index}" for index in range(1, 13)]]
    model_layers = {model: layers for model in ALL_SCOPE_MODELS}
    destination = tmp_path / "deadline_scope"

    payload = publish_compute_scope_manifests(
        destination,
        model_layers=model_layers,
        hubert_original_units={layer: f"/preserved/{layer}" for layer in layers},
    )

    assert payload["task_counts"] == {
        "primary_fast": 8,
        "controls": 21,
        "nulls": 80,
    }
    assert payload["resolved_depth_layers"]["hubert_base"]["middle"] == "encoder_6"
    primary = pd.read_csv(destination / "primary_fast_refits.tsv", sep="\t")
    controls = pd.read_csv(destination / "selected_depth_controls.tsv", sep="\t")
    nulls = pd.read_csv(destination / "structured_nulls.tsv", sep="\t")
    assert len(primary) == 8
    assert len(controls) == 21
    assert len(nulls) == 80
    assert set(nulls["layer"]) == {"encoder_6"}
    assert set(nulls["shift"]) == set(range(20))
    assert publish_compute_scope_manifests(
        destination,
        model_layers=model_layers,
        hubert_original_units={layer: f"/preserved/{layer}" for layer in layers},
    ) == payload
    saved = json.loads((destination / "resolved_model_layers.json").read_text())
    assert saved["hubert_base_original"]["action"].endswith("do_not_rerun")
    assert validate_selected_depth_controls(
        destination, model_layers=model_layers
    ) == {
        "state": "valid",
        "model_count": 9,
        "task_counts": {
            "primary_fast": 8,
            "controls": 21,
            "nulls": 80,
        },
    }

    changed = dict(model_layers)
    changed["hubert_base"] = ["input", "different_middle", "different_final"]
    with pytest.raises(ValueError, match="manifest is stale"):
        validate_selected_depth_controls(destination, model_layers=changed)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_compute_scope_manifests(
            destination,
            model_layers=changed,
            hubert_original_units={"input": "/preserved/input"},
        )
    refreshed = publish_compute_scope_manifests(
        destination,
        model_layers=changed,
        hubert_original_units={"input": "/preserved/input"},
        preserve_stale=True,
    )
    assert refreshed["resolved_depth_layers"]["hubert_base"] == {
        "input": "input",
        "middle": "different_middle",
        "final": "different_final",
    }
    assert len(list(tmp_path.glob("deadline_scope.stale-*"))) == 1


def test_selected_depth_validation_preserves_model_specific_layers(tmp_path):
    model_layers = {
        model: ["input", f"{model}_middle", f"{model}_final"]
        for model in ALL_SCOPE_MODELS
    }
    destination = tmp_path / "model_specific_scope"
    publish_compute_scope_manifests(
        destination,
        model_layers=model_layers,
        hubert_original_units={"input": "/preserved/input"},
    )

    result = validate_selected_depth_controls(
        destination, model_layers=model_layers
    )
    controls = pd.read_csv(destination / "selected_depth_controls.tsv", sep="\t")

    assert result["state"] == "valid"
    assert result["task_counts"]["controls"] == 21
    for row in controls.itertuples(index=False):
        assert [row.input, row.middle, row.final] == model_layers[row.model]


def test_refresh_archives_corrupt_task_tsv_even_when_json_is_current(tmp_path):
    model_layers = {
        model: ["input", f"{model}_middle", f"{model}_final"]
        for model in ALL_SCOPE_MODELS
    }
    destination = tmp_path / "scope"
    kwargs = {
        "model_layers": model_layers,
        "hubert_original_units": {"input": "/preserved/input"},
    }
    publish_compute_scope_manifests(destination, **kwargs)
    tasks = destination / "selected_depth_controls.tsv"
    tasks.write_text(
        tasks.read_text().replace("hubert_base_middle", "absent_layer", 1)
    )

    with pytest.raises(FileExistsError, match="invalid manifests"):
        publish_compute_scope_manifests(destination, **kwargs)
    publish_compute_scope_manifests(
        destination,
        preserve_stale=True,
        **kwargs,
    )

    assert validate_selected_depth_controls(
        destination, model_layers=model_layers
    )["state"] == "valid"
    assert len(list(tmp_path.glob("scope.stale-*"))) == 1


def test_manifest_writer_uses_lf_and_validator_rejects_crlf(tmp_path):
    model_layers = {
        model: ["input", f"{model}_middle", f"{model}_final"]
        for model in ALL_SCOPE_MODELS
    }
    destination = tmp_path / "scope"
    publish_compute_scope_manifests(
        destination,
        model_layers=model_layers,
        hubert_original_units={"input": "/preserved/input"},
    )
    tasks = destination / "selected_depth_controls.tsv"
    canonical = tasks.read_bytes()
    assert b"\r" not in canonical
    assert canonical.endswith(b"\n")

    tasks.write_bytes(canonical.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="canonical LF"):
        validate_selected_depth_controls(
            destination,
            model_layers=model_layers,
        )
