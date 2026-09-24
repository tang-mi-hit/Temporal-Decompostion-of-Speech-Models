from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from speech_strf.stage4_audit import EXPECTED_MODEL_IDENTITIES, STAGE4_MODEL_DIRECTORIES
from speech_strf.stage4_runner import Stage4Runner, load_stage4_config, validate_unit


def _config(tmp_path: Path) -> Path:
    source = load_stage4_config(
        Path(__file__).parents[1] / "configs" / "stage4_revision.yaml"
    )
    source["manifest"] = "outputs/manifest.csv"
    source["model_root"] = "outputs"
    source["audit_output"] = "outputs/stage4_revision/audit/input_audit.json"
    source["features"] = {
        "original": "outputs/features",
        "rich": "outputs/features",
    }
    source["output_root"] = "outputs/stage4_revision"
    source["expected_recording_count"] = 4
    source["governance_inputs"] = {
        "revision_roadmap": "governance/revision_roadmap.json",
        "claim_surface_manifest": "governance/claim_surface_manifest.json",
    }
    source["encoding"] = {
        "alphas": [1.0],
        "target_pca_components": None,
        "sensitivity_folds": None,
        "inner_folds": 3,
    }
    config = tmp_path / "configs" / "stage4_revision.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    for name in ("data.yaml", "models.yaml", "features.yaml", "analysis.yaml"):
        (config.parent / name).write_text("{}\n", encoding="utf-8")
    return config


def _inputs(tmp_path: Path, *, frames: int = 12) -> None:
    governance = tmp_path / "governance"
    governance.mkdir()
    (governance / "revision_roadmap.json").write_text("{}")
    (governance / "claim_surface_manifest.json").write_text("{}")
    outputs = tmp_path / "outputs"
    features = outputs / "features"
    features.mkdir(parents=True)
    ids = [f"recording_{index}" for index in range(4)]
    pd.DataFrame({"recording_id": ids}).to_csv(outputs / "manifest.csv", index=False)
    rng = np.random.default_rng(3)
    times = np.arange(frames) / 50
    coefficient = rng.normal(size=(5, 3))
    matrices = {}
    targets = {}
    for recording_id in ids:
        matrix = rng.normal(size=(frames, 5))
        target = matrix @ coefficient
        matrices[recording_id] = matrix
        targets[recording_id] = target
        np.savez_compressed(
            features / f"{recording_id}.npz",
            matrix=matrix,
            times=times,
            names=np.array(["a", "p", "ph", "w", "o"]),
            families=np.array(
                ["acoustic", "prosodic", "phonetic", "word", "onset"]
            ),
        )

    for model in STAGE4_MODEL_DIRECTORIES:
        root = outputs / model
        root.mkdir()
        identity = EXPECTED_MODEL_IDENTITIES[model]
        for filename in (
            "run_metadata.json",
            "layer_metadata.json",
            "comparability_contract.json",
        ):
            payload = {} if filename.startswith("comparability") else {
                "model": {"key": identity}
            }
            (root / filename).write_text(json.dumps(payload), encoding="utf-8")
        pd.DataFrame(
            {"recording_id": ids, "duration_seconds": [frames / 50] * len(ids)}
        ).to_csv(root / "extraction_manifest.csv", index=False)
        with h5py.File(root / "activations.h5", "w") as store:
            for recording_id in ids:
                group = store.create_group(recording_id)
                group.attrs["complete"] = True
                group.attrs["layer_names_json"] = json.dumps(["layer_00_input"])
                group.attrs["metadata_json"] = json.dumps(
                    {
                        "model_key": identity,
                        "native_frame_count": len(times),
                        "canonical_frame_count": len(times),
                    }
                )
                group.create_dataset("native_timestamps", data=times)
                group.create_dataset("canonical_timestamps", data=times)
                group.create_group("native").create_dataset(
                    "layer_00_input", data=targets[recording_id]
                )
                group.create_group("canonical").create_dataset(
                    "layer_00_input", data=targets[recording_id]
                )


def test_synthetic_end_to_end_is_deterministic_and_writes_under_stage4(tmp_path):
    config = _config(tmp_path)
    runner = Stage4Runner(config)

    first = runner.synthetic_test()
    second = runner.synthetic_test()

    for key in ("passed", "recording_count", "score_rows", "seed"):
        assert first[key] == second[key]
    assert first["passed"]
    assert first["runtime_seconds"] >= 0
    assert first["peak_rss_kib"] > 0
    result_path = (
        tmp_path / "outputs" / "stage4_revision" / "synthetic_test" / "result.json"
    )
    assert json.loads(result_path.read_text()) == second


def test_functional_smoke_uses_one_real_recording_and_one_null_shift(tmp_path):
    config = _config(tmp_path)
    _inputs(tmp_path, frames=220)
    runner = Stage4Runner(config)

    result = runner.functional_smoke("hubert_base")

    assert result["state"] == "complete"
    assert result["recordings_loaded"] == 1
    assert result["null_shift_count"] == 1
    assert result["alpha_grid"] == [1.0]
    assert result["synthetic_nested_cv"]["passed"]
    assert result["real_recording_nested_cv"]["state"] == "not_run"
    status_paths = list(
        (
            tmp_path
            / "outputs"
            / "stage4_revision"
            / "functional_smoke"
            / "hubert_base"
        ).glob("*/*/status.json")
    )
    assert len(status_paths) == 1
    status = json.loads(status_paths[0].read_text(encoding="utf-8"))
    assert set(status["null_shift_manifest"]["recordings"]) == {
        result["recording_id"]
    }


def test_corrupt_completed_unit_is_preserved_diagnosed_and_recomputed(tmp_path):
    config = _config(tmp_path)
    _inputs(tmp_path)
    runner = Stage4Runner(config)

    first = runner.fit("hubert_base", "zero_lag", "layer_00_input")
    unit = Path(first[0]["path"])
    valid, reason = validate_unit(unit)
    assert valid, reason
    (unit / "scores.csv").write_text("corrupt\n", encoding="utf-8")

    second = runner.fit("hubert_base", "zero_lag", "layer_00_input")

    assert second[0]["state"] == "computed"
    preserved = Path(second[0]["preserved_corrupt"])
    assert preserved.is_dir()
    diagnostic = json.loads(
        (preserved / "corruption_diagnostic.json").read_text(encoding="utf-8")
    )
    assert "artifact_hash_mismatch:scores.csv" in diagnostic["reason"]
    valid, reason = validate_unit(unit)
    assert valid, reason
    scores = pd.read_csv(unit / "scores.csv")
    assert set(scores["model"]) == {"hubert_base"}
    assert set(scores["layer"]) == {"layer_00_input"}
    assert set(scores["variant"]) == {"zero_lag"}


def test_null_summary_refuses_any_missing_model_layer_ensemble(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _inputs(tmp_path)
    runner = Stage4Runner(config)
    rows = []
    for model in STAGE4_MODEL_DIRECTORIES:
        for recording_id in [f"recording_{index}" for index in range(4)]:
            for family in ("acoustic", "prosodic", "phonetic", "word", "onset"):
                rows.append(
                    {
                        "model": model,
                        "layer": "layer_00_input",
                        "recording_id": recording_id,
                        "family": family,
                        "split_kind": "primary",
                        "delta_r2": 0.1,
                        "duration_seconds": 12 / 50,
                    }
                )
    monkeypatch.setattr(
        runner, "_completed_fit_frames", lambda *args, **kwargs: [pd.DataFrame(rows)]
    )
    monkeypatch.setattr(
        runner, "_source_hashes", lambda *args, **kwargs: {"source": "hash"}
    )
    (
        tmp_path / "outputs" / "stage4_revision" / "model_comparison"
    ).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="null set is incomplete"):
        runner._summarize_nulls(
            [f"recording_{index}" for index in range(4)], 10_000, 17
        )
