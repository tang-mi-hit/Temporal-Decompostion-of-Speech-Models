from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from speech_strf.stage4_compute_scope import (
    ALL_SCOPE_MODELS,
    publish_compute_scope_manifests,
)
from speech_strf.stage4_handoff import (
    build_stage4_handoff,
    verify_stage4_handoff,
)
from speech_strf.stage4_runner import Stage4Runner
from test_stage4_runner import _config, _inputs


def _handoff_fixture(tmp_path: Path) -> tuple[Path, Stage4Runner]:
    config = _config(tmp_path)
    _inputs(tmp_path)
    source_rich = (
        Path(__file__).parents[1] / "configs" / "features_stage4_rich.yaml"
    )
    (tmp_path / "configs" / "features_stage4_rich.yaml").write_bytes(
        source_rich.read_bytes()
    )
    runner = Stage4Runner(config)
    result = runner.fit("hubert_base", "original", "layer_00_input")
    unit = Path(result[0]["path"])
    layers = {model: ["layer_00_input", "layer_01_transformer", "layer_02_transformer"] for model in ALL_SCOPE_MODELS}
    manifest_root = (
        tmp_path
        / "outputs"
        / "stage4_revision"
        / "manifests"
        / "deadline_compute_scope"
    )
    publish_compute_scope_manifests(
        manifest_root,
        model_layers=layers,
        hubert_original_units={"layer_00_input": str(unit)},
    )
    qc = tmp_path / "outputs" / "stage4_revision" / "provenance_qc"
    qc.mkdir()
    (qc / "provenance_qc.json").write_text(
        json.dumps(
            {
                "status": "WARN",
                "reasons": [
                    {"status": "WARN", "reason": "synthetic_qc_warning"}
                ],
            }
        )
    )
    (qc / "temporal_context.csv").write_text("model,status\nhubert_base,PASS\n")
    logs = tmp_path / "outputs" / "stage4_revision" / "logs"
    logs.mkdir()
    (logs / "tests.xml").write_text("<testsuite tests='1' failures='0'/>\n")
    return config, runner


def test_partial_handoff_is_atomic_hashed_and_excludes_large_inputs(tmp_path):
    config, runner = _handoff_fixture(tmp_path)

    report = build_stage4_handoff(config)
    handoff = runner.output_root / "handoff"

    assert report["state"] == "PARTIAL"
    assert report["missing_unit_count"] > 0
    assert verify_stage4_handoff(handoff)["state"] == "valid"
    assert (handoff / "tables" / "primary_recording_level.csv").is_file()
    primary = pd.read_csv(handoff / "tables" / "primary_recording_level.csv")
    assert set(primary["model"]) == {"hubert_base"}
    assert set(primary["recording_id"]) == {
        f"recording_{index}" for index in range(4)
    }
    assert (handoff / "verification" / "missing_incomplete_units.csv").is_file()
    assert (
        handoff
        / "provenance_qc"
        / "model_metadata"
        / "hubert_base"
        / "run_metadata.json"
    ).is_file()
    assert (handoff / "verification" / "governance").is_dir()
    assert len(list(handoff.rglob("predictions.npz"))) == 1
    assert not list(handoff.rglob("activations.h5"))
    assert not list(handoff.rglob("features_rich"))
    sums = (handoff / "SHA256SUMS").read_text()
    assert "HANDOFF.json" in sums
    assert "primary_recording_level.csv" in sums

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_stage4_handoff(config)


def test_handoff_verifier_rejects_bulk_predictions(tmp_path):
    config, runner = _handoff_fixture(tmp_path)
    build_stage4_handoff(config)
    handoff = runner.output_root / "handoff"
    forbidden = handoff / "tables" / "predictions.npz"
    forbidden.write_bytes(b"not-a-real-prediction")

    with pytest.raises(ValueError, match="SHA256SUMS coverage mismatch"):
        verify_stage4_handoff(handoff)
