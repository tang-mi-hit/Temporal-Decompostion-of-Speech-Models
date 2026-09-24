"""Regression coverage for Stage 4 TSV-to-Slurm layer arguments."""

from __future__ import annotations

import json
import subprocess

import h5py

from speech_strf.stage4_compute_scope import (
    ALL_SCOPE_MODELS,
    publish_compute_scope_manifests,
    validate_compute_scope_manifests,
)
from speech_strf.stage4_runner import Stage4Runner
from test_stage4_runner import _config, _inputs


def test_lf_final_layer_round_trips_through_awk_read_and_process_fit(tmp_path):
    config = _config(tmp_path)
    _inputs(tmp_path)
    layers = [
        "layer_00_input",
        "layer_01_transformer",
        "layer_02_transformer",
    ]
    activation_path = tmp_path / "outputs" / "hubert_base" / "activations.h5"
    with h5py.File(activation_path, "r+") as store:
        for group in store.values():
            values = group["canonical/layer_00_input"][...]
            for layer in layers[1:]:
                group["native"].create_dataset(layer, data=values)
                group["canonical"].create_dataset(layer, data=values)
            group.attrs["layer_names_json"] = json.dumps(layers)

    manifest_dir = tmp_path / "scope"
    model_layers = {model: layers for model in ALL_SCOPE_MODELS}
    publish_compute_scope_manifests(
        manifest_dir,
        model_layers=model_layers,
        hubert_original_units={layer: f"/preserved/{layer}" for layer in layers},
    )
    task_bytes = (manifest_dir / "selected_depth_controls.tsv").read_bytes()
    assert b"\r" not in task_bytes
    assert task_bytes.endswith(b"\n")
    assert validate_compute_scope_manifests(
        manifest_dir, model_layers=model_layers
    )["state"] == "valid"
    Stage4Runner(config)._audit()

    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""
set -euo pipefail
TASKS=$1
CONFIG=$2
SLURM_ARRAY_TASK_ID=0
IFS=$'\t' read -r MODEL VARIANT INPUT MIDDLE FINAL < <(
  awk -F $'\t' -v task="$SLURM_ARRAY_TASK_ID" \
    'NR > 1 && $1 == task {print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6}' "$TASKS"
)
/usr/bin/python3 scripts/run_stage4_revision.py \
  --config "$CONFIG" fit "$MODEL" "$VARIANT" \
  --layers "$INPUT" "$MIDDLE" "$FINAL" --layer-workers 2
""",
            "bash",
            str(manifest_dir / "selected_depth_controls.tsv"),
            str(config),
        ],
        cwd="/workspace/project",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "layer_02_transformer" in result.stdout
