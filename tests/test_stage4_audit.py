import json

import h5py
import numpy as np
import pandas as pd

from speech_strf.stage4_audit import (
    EXPECTED_MODEL_IDENTITIES,
    STAGE4_MODEL_DIRECTORIES,
    audit_stage4_inputs,
)


def _write_complete_inputs(tmp_path):
    outputs = tmp_path / "outputs"
    features = outputs / "features"
    features.mkdir(parents=True)
    recording_id = "story01"
    times = np.array([0.0, 0.02, 0.04])
    pd.DataFrame({"recording_id": [recording_id]}).to_csv(
        outputs / "manifest.csv", index=False
    )
    np.savez_compressed(
        features / f"{recording_id}.npz",
        matrix=np.ones((3, 2), dtype=np.float32),
        times=times,
        names=np.array(["a", "b"]),
        families=np.array(["acoustic", "word"]),
    )
    for directory in STAGE4_MODEL_DIRECTORIES:
        root = outputs / directory
        root.mkdir()
        identity = EXPECTED_MODEL_IDENTITIES[directory]
        (root / "run_metadata.json").write_text(
            json.dumps({"model": {"key": identity}})
        )
        (root / "layer_metadata.json").write_text(
            json.dumps({"model": {"key": identity}})
        )
        (root / "comparability_contract.json").write_text(json.dumps({}))
        pd.DataFrame(
            {"recording_id": [recording_id], "duration_seconds": [0.06]}
        ).to_csv(root / "extraction_manifest.csv", index=False)
        with h5py.File(root / "activations.h5", "w") as store:
            group = store.create_group(recording_id)
            group.attrs["complete"] = True
            group.attrs["layer_names_json"] = json.dumps(["layer_00_input"])
            group.attrs["metadata_json"] = json.dumps(
                {
                    "model_key": identity,
                    "native_frame_count": 3,
                    "canonical_frame_count": 3,
                }
            )
            group.create_dataset("native_timestamps", data=times)
            group.create_dataset("canonical_timestamps", data=times)
            group.create_group("native").create_dataset(
                "layer_00_input", data=np.ones((3, 2), dtype=np.float32)
            )
            group.create_group("canonical").create_dataset(
                "layer_00_input", data=np.ones((3, 2), dtype=np.float32)
            )
    return outputs


def test_complete_audit_resolves_hubert_rerun_identity(tmp_path):
    outputs = _write_complete_inputs(tmp_path)
    destination = outputs / "stage4_revision" / "audit" / "input_audit.json"

    report = audit_stage4_inputs(
        outputs / "manifest.csv", outputs / "features", outputs, destination
    )

    assert report["complete"]
    assert destination.is_file()
    persisted = json.loads(destination.read_text())
    assert persisted["complete"]
    rerun = next(
        model
        for model in report["models"]
        if model["directory"] == "hubert_large_refactor_rerun"
    )
    assert rerun["resolved_model_identity"] == "hubert_large_reference"
    assert rerun["path"].endswith("hubert_large_refactor_rerun")
    assert len(report["recordings"][0]["models"]) == 9


def test_incomplete_hdf5_group_fails_but_writes_audit(tmp_path):
    outputs = _write_complete_inputs(tmp_path)
    store_path = outputs / STAGE4_MODEL_DIRECTORIES[0] / "activations.h5"
    with h5py.File(store_path, "a") as store:
        store["story01"].attrs["complete"] = False
    destination = outputs / "stage4_revision" / "audit" / "input_audit.json"

    report = audit_stage4_inputs(
        outputs / "manifest.csv", outputs / "features", outputs, destination
    )

    assert not report["complete"]
    assert any("HDF5 group is not marked complete" in error for error in report["errors"])
    assert json.loads(destination.read_text())["complete"] is False
