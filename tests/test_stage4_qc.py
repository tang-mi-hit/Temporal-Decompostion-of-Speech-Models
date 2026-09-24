import json

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from speech_strf.stage4_audit import (
    EXPECTED_MODEL_IDENTITIES,
    STAGE4_MODEL_DIRECTORIES,
)
from speech_strf.stage4_qc import LAG_CONVENTION, _annotation_qc, run_stage4_qc


def _textgrid() -> str:
    return """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "ma1"
        intervals [2]:
            xmin = 0.5
            xmax = 1
            text = "ma2"
    item [2]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "妈妈"
"""


def _write_sources(tmp_path, *, explicit_context=True):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    audio = inputs / "story.wav"
    alignment = inputs / "story.TextGrid"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    alignment.write_text(_textgrid(), encoding="utf-8")
    manifest = outputs / "manifest.csv"
    pd.DataFrame(
        {
            "recording_id": ["story"],
            "audio_path": [str(audio)],
            "alignment_path": [str(alignment)],
        }
    ).to_csv(manifest, index=False)

    for directory in STAGE4_MODEL_DIRECTORIES:
        root = outputs / directory
        root.mkdir()
        identity = EXPECTED_MODEL_IDENTITIES[directory]
        run_metadata = {
            "model": {
                "key": identity,
                "model_id": f"org/{identity}",
                "revision": "main",
            },
            "resolved_model_revision": "a" * 40,
            "package_versions": {"transformers": "1.2.3"},
            "git_commit": "b" * 40,
        }
        (root / "run_metadata.json").write_text(json.dumps(run_metadata))
        metadata = {
            "model_key": identity,
            "model_id": f"org/{identity}",
            "requested_revision": "main",
            "resolved_revision": "a" * 40,
            "input_sample_rate_hz": 16000,
            "native_frame_rate_hz": 50.0,
            "canonical_rate_hz": 50,
            "processor_identity": f"org/{identity}",
            "config_identity": "config-v1",
            "layers": [{"name": "layer_00_input", "hidden_size": 2}],
            "extraction": {
                "batch_seconds_effective": 20.0,
                "overlap_seconds_effective": 1.0,
                "stitching": (
                    "retain_frame_centers_within_nonoverlapping_core_windows"
                ),
                "frame_stride_samples": 320,
            },
            "extraction_signature": {
                "adapter": "generic_speech",
                "local_checkpoint_sha256": "c" * 64,
            },
        }
        if explicit_context:
            metadata.update(
                {
                    "bidirectional_context": True,
                    "future_context_seconds": 1.0,
                    "receptive_field_samples": 400,
                }
            )
        (root / "layer_metadata.json").write_text(
            json.dumps(
                {
                    "model": {"key": identity, "model_id": f"org/{identity}"},
                    "layers": metadata["layers"],
                    "observed_model_metadata": metadata,
                }
            )
        )
        (root / "comparability_contract.json").write_text(
            json.dumps({"canonical_rate_hz": 50})
        )
        pd.DataFrame(
            {
                "recording_id": ["story"],
                "duration_seconds": [1.0],
                "metadata_json": [json.dumps(metadata)],
            }
        ).to_csv(root / "extraction_manifest.csv", index=False)
        with h5py.File(root / "activations.h5", "w") as store:
            group = store.create_group("story")
            group.attrs["complete"] = True
            group.attrs["metadata_json"] = json.dumps(metadata)
            group.attrs["layer_names_json"] = json.dumps(["layer_00_input"])
            group.create_dataset("native_timestamps", data=[0.01, 0.03])
            group.create_dataset("canonical_timestamps", data=[0.0, 0.02])
            group.create_group("native").create_dataset(
                "layer_00_input", data=np.ones((2, 2))
            )
            group.create_group("canonical").create_dataset(
                "layer_00_input", data=np.ones((2, 2))
            )
    return manifest, outputs


def test_stage4_qc_writes_provenance_annotation_and_context_outputs(tmp_path):
    manifest, outputs = _write_sources(tmp_path)
    destination = outputs / "stage4_revision" / "provenance_qc"
    registry = tmp_path / "models.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": {
                    identity: {"model_id": f"registered/{identity}"}
                    for identity in EXPECTED_MODEL_IDENTITIES.values()
                }
            }
        )
    )

    report = run_stage4_qc(
        manifest, outputs, destination, model_config_path=registry
    )

    assert report["status"] == "PASS"
    assert report["model_directories"] == list(STAGE4_MODEL_DIRECTORIES)
    assert len(report["models"]) == 9
    assert report["models"][0]["checkpoint_id"].startswith("org/")
    assert report["models"][0]["registered_hf_model_id"].startswith("registered/")
    assert report["models"][0]["checkpoint_fingerprint"] == "c" * 64
    assert report["models"][0]["processor_identity"].startswith("org/")
    assert report["annotation_totals"]["phone_feature_family"] == "phonetic"
    assert report["annotation_totals"]["phone_category_distribution"] == {
        "ma1": 1,
        "ma2": 1,
    }
    assert report["recordings"][0]["word_count"] == 1
    assert report["temporal_context"][0]["context_audit_support"] == "supported"
    assert report["lag_convention"] == LAG_CONVENTION
    assert {
        "provenance_qc.json",
        "provenance_qc.csv",
        "recording_annotation_qc.csv",
        "temporal_context.csv",
        "annotation_totals.csv",
    } <= {path.name for path in destination.iterdir()}
    persisted = json.loads((destination / "provenance_qc.json").read_text())
    assert persisted["lag_convention"] == LAG_CONVENTION


def test_temporal_context_does_not_infer_unsaved_fields(tmp_path):
    manifest, outputs = _write_sources(tmp_path, explicit_context=False)

    report = run_stage4_qc(
        manifest, outputs, outputs / "stage4_revision" / "provenance_qc"
    )

    context = report["temporal_context"][0]
    assert context["context_audit_support"] == "unsupported"
    assert context["bidirectional_context"] == "unknown"
    assert context["future_context"] == "unknown"
    assert context["receptive_field"] == "unknown"


def test_missing_required_model_source_fails_precisely(tmp_path):
    manifest, outputs = _write_sources(tmp_path)
    missing = outputs / STAGE4_MODEL_DIRECTORIES[3] / "run_metadata.json"
    missing.unlink()

    try:
        run_stage4_qc(
            manifest, outputs, outputs / "stage4_revision" / "provenance_qc"
        )
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing required source did not fail")

    assert STAGE4_MODEL_DIRECTORIES[3] in message
    assert str(missing) in message


def test_sub_microsecond_labeled_endpoint_quantization_is_not_fatal(tmp_path):
    audio = tmp_path / "quantized.wav"
    alignment = tmp_path / "quantized.TextGrid"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    alignment.write_text(
        _textgrid().replace("xmax = 1", "xmax = 1.0000005"),
        encoding="utf-8",
    )

    result = _annotation_qc("quantized", audio, alignment, 0.03)

    assert result["out_of_bounds_count"] == 2
    assert result["fatal_out_of_bounds_count"] == 0
    assert result["tolerated_numerical_out_of_bounds_count"] == 2
    assert result["status"] == "WARN"


def test_labeled_endpoint_overhang_above_numerical_tolerance_is_fatal(tmp_path):
    audio = tmp_path / "overhang.wav"
    alignment = tmp_path / "overhang.TextGrid"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    alignment.write_text(
        _textgrid().replace("xmax = 1", "xmax = 1.000002"),
        encoding="utf-8",
    )

    result = _annotation_qc("overhang", audio, alignment, 0.03)

    assert result["out_of_bounds_count"] == 2
    assert result["tolerated_numerical_out_of_bounds_count"] == 0
    assert result["fatal_out_of_bounds_count"] == 2
    assert result["status"] == "FAIL"

